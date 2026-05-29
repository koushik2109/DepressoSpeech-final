"""Async SQLAlchemy engine and session factory for PostgreSQL."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config.settings import get_settings

settings = get_settings()

# PostgreSQL RDBMS — use a proper connection pool
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
