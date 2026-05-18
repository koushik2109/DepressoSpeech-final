"""Async SQLAlchemy engine and session factory."""

from sqlalchemy import event
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config.settings import get_settings

settings = get_settings()

_is_sqlite = "sqlite" in settings.DATABASE_URL
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

if _is_sqlite:
    # SQLite + aiosqlite uses NullPool — pooling is meaningless for a local file
    from sqlalchemy.pool import NullPool
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        connect_args=_connect_args,
        poolclass=NullPool,
    )
else:
    # PostgreSQL / other RDBMS — use a proper connection pool
    # Pass ssl=False for local/dev PostgreSQL (asyncpg attempts SSL by default)
    _pg_connect_args = {}
    _pg_url = settings.DATABASE_URL
    if "127.0.0.1" in _pg_url or "localhost" in _pg_url:
        _pg_connect_args["ssl"] = False
    engine = create_async_engine(
        _pg_url,
        echo=settings.DATABASE_ECHO,
        connect_args=_pg_connect_args,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )

# Enable WAL mode for SQLite: allows concurrent reads alongside writes,
# dramatically reducing lock contention under parallel API requests.
if "sqlite" in settings.DATABASE_URL:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _conn_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-32000")  # 32 MB page cache
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI dependency – yields an async DB session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Create all tables (import models first so they register with Base)."""
    import src.models  # noqa: F401 – registers models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if "sqlite" in settings.DATABASE_URL:
            await _run_sqlite_migrations(conn)


async def _run_sqlite_migrations(conn):
    async def _columns(table_name: str) -> set[str]:
        result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
        rows = result.fetchall()
        return {row[1] for row in rows}

    async def _add_column(table_name: str, column_name: str, column_sql: str) -> None:
        existing = await _columns(table_name)
        if column_name not in existing:
            await conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"))

    await _add_column("doctors", "patient_count", "patient_count INTEGER NOT NULL DEFAULT 0")
    await _add_column("assessments", "report_status", "report_status VARCHAR(16) NOT NULL DEFAULT 'pending'")
    await _add_column("assessments", "is_report_ready", "is_report_ready BOOLEAN NOT NULL DEFAULT 0")
    await _add_column("assessments", "doctor_remarks", "doctor_remarks TEXT")

    await conn.execute(
        text(
            """
            UPDATE doctors
            SET patient_count = (
                SELECT COUNT(DISTINCT da.patient_id)
                FROM doctor_assignments da
                WHERE da.doctor_id = doctors.id
                  AND da.status IN ('accepted', 'completed')
            )
            """
        )
    )
    await conn.execute(
        text(
            """
            UPDATE assessments
            SET
                report_status = CASE
                    WHEN status = 'completed' THEN 'available'
                    ELSE COALESCE(report_status, 'pending')
                END,
                is_report_ready = CASE
                    WHEN status = 'completed' THEN 1
                    ELSE COALESCE(is_report_ready, 0)
                END
            """
        )
    )
    await conn.execute(
        text(
            """
            UPDATE assessments
            SET
                status = 'completed',
                report_status = 'available',
                is_report_ready = 1
            WHERE id IN (
                SELECT assessment_id
                FROM processing_jobs
                WHERE assessment_id IS NOT NULL
                  AND status = 'succeeded'
                  AND progress_pct >= 100
            )
            """
        )
    )

    # Migrate existing doctor_assignments into consultations on first boot
    count_result = await conn.execute(text("SELECT COUNT(*) FROM consultations"))
    consultation_count = count_result.scalar()
    if consultation_count == 0:
        await conn.execute(
            text(
                """
                INSERT INTO consultations (
                    id, doctor_id, patient_id, assessment_id, doctor_assignment_id,
                    status, created_at, updated_at, started_at, ended_at, stop_reason
                )
                SELECT
                    id, doctor_id, patient_id, assessment_id, id,
                    CASE
                        WHEN status = 'accepted' THEN 'active'
                        WHEN status = 'completed' THEN 'completed'
                        WHEN status = 'rejected' THEN 'rejected'
                        ELSE 'pending'
                    END,
                    created_at, created_at,
                    CASE WHEN status = 'accepted' THEN created_at ELSE NULL END,
                    CASE WHEN status IN ('completed', 'rejected') THEN created_at ELSE NULL END,
                    NULL
                FROM doctor_assignments
                """
            )
        )
