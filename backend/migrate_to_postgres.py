"""
SQLite → PostgreSQL migration script
=====================================
Copies every row from the local mindscope.db (SQLite) into the
PostgreSQL database configured in DATABASE_URL, then migrates any
locally-stored audio/video files into the media_file_data BYTEA table.

Usage
-----
1.  Make sure backend/.env has:
        DATABASE_URL=postgresql+asyncpg://mindscope:Mindscope@2026!@localhost:5432/mindscope
        STORAGE_PROVIDER=postgres

2.  Run from the backend directory:
        python migrate_to_postgres.py

The script is IDEMPOTENT: re-running it skips rows that already exist.
"""

import asyncio
import logging
import sys
from pathlib import Path

# ── bootstrap the path so backend imports work ────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("migrate")


# ── helpers ────────────────────────────────────────────────────────────────────

def _sqlite_rows(sqlite_url: str, table: str) -> list[dict]:
    """Return all rows from a SQLite table as dicts."""
    import sqlite3
    raw = sqlite_url.replace("sqlite+aiosqlite:///", "")
    p = Path(raw)
    # If the path is absolute (starts with /), use it as-is; else resolve relative to BACKEND_DIR
    db_path = p if p.is_absolute() else (BACKEND_DIR / raw).resolve()
    if not db_path.exists():
        logger.warning("SQLite DB not found at %s — skipping table %s", db_path, table)
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(f"SELECT * FROM {table}")
        rows = [dict(r) for r in cur.fetchall()]
        logger.info("  SQLite %-35s → %d rows", table, len(rows))
        return rows
    except sqlite3.OperationalError as exc:
        logger.warning("  Skipping %s: %s", table, exc)
        return []
    finally:
        conn.close()


async def _get_col_types(pg_engine, table: str) -> dict[str, str]:
    """Return {column_name: data_type} for the given PG table."""
    from sqlalchemy import text
    async with pg_engine.connect() as conn:
        result = await conn.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = :t"
        ), {"t": table})
        return {row[0]: row[1] for row in result.fetchall()}


def _coerce_row(row: dict, col_types: dict[str, str]) -> dict:
    """Convert SQLite types to proper Python types for PostgreSQL asyncpg.

    SQLite quirks:
    - Booleans stored as 0/1 integers
    - Datetimes stored as strings like '2026-04-15 18:37:39.064482'
    """
    from datetime import datetime, timezone
    out = {}
    for k, v in row.items():
        col_type = col_types.get(k, "")
        if v is None:
            out[k] = None
        elif col_type == "boolean" and isinstance(v, int):
            out[k] = bool(v)
        elif "timestamp" in col_type and isinstance(v, str):
            # Parse ISO-like datetime string from SQLite
            try:
                dt = datetime.fromisoformat(v)
                # Make timezone-aware if the column is TIMESTAMPTZ
                if "with time zone" in col_type and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                out[k] = dt
            except ValueError:
                out[k] = v  # Leave as-is if unparseable
        else:
            out[k] = v
    return out


async def _upsert(pg_engine, table: str, rows: list[dict], pk: str = "id") -> int:
    """Insert rows; skip any that already exist (ON CONFLICT DO NOTHING)."""
    if not rows:
        return 0
    from sqlalchemy import text
    col_types = await _get_col_types(pg_engine, table)
    cols = list(rows[0].keys())
    col_list = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    stmt = text(
        f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders}) '
        f'ON CONFLICT ("{pk}") DO NOTHING'
    )
    from sqlalchemy.exc import IntegrityError
    inserted = 0
    skipped = 0
    for row in rows:
        coerced = _coerce_row(row, col_types)
        try:
            async with pg_engine.begin() as conn:
                result = await conn.execute(stmt, coerced)
                inserted += result.rowcount
        except IntegrityError as exc:
            # Skip rows with broken FK references (data that SQLite allowed but PG rejects)
            logger.debug("  Skipping row in %s (integrity violation): %s", table, str(exc.orig)[:120])
            skipped += 1
    if skipped:
        logger.warning("  %-35s skipped %d rows (orphaned FK references)", table, skipped)
    return inserted


# ── ordered table list (FK order) ─────────────────────────────────────────────

TABLES_IN_ORDER = [
    "users",
    "doctors",
    "doctor_assignments",
    "assessments",
    "media_files",           # must precede assessment_answers (FK: audio_file_id)
    "media_file_data",       # BYTEA data for media_files
    "assessment_answers",
    "assessment_ml_details",
    "processing_jobs",
    "multimodal_sessions",
    "consultations",
    "request_metrics",
    "otps",
]

# Tables whose primary key is not 'id'
_PK_OVERRIDES = {
    "media_file_data": "file_id",
}


# ── main migration ─────────────────────────────────────────────────────────────

async def main() -> None:
    from config.settings import get_settings
    settings = get_settings()

    if "sqlite" in settings.DATABASE_URL:
        logger.error(
            "DATABASE_URL still points to SQLite (%s).\n"
            "Edit backend/.env and set DATABASE_URL to your PostgreSQL URL first.",
            settings.DATABASE_URL,
        )
        sys.exit(1)

    # Prefer backend/mindscope.db (has all application data).
    # Fall back to repo-root mindscope.db only when the backend one is absent.
    _local_db = BACKEND_DIR / "mindscope.db"
    _root_db  = BACKEND_DIR.parent / "mindscope.db"
    _sqlite_path = _local_db if _local_db.exists() else _root_db
    sqlite_url = f"sqlite+aiosqlite:///{_sqlite_path.as_posix()}"
    logger.info("SQLite source: %s (exists=%s)", _sqlite_path, _sqlite_path.exists())

    # ── 1. Create all PG tables ────────────────────────────────────────────────
    logger.info("Step 1 — Creating PostgreSQL schema …")
    from database.base import engine, init_db
    await init_db()
    logger.info("  Schema ready.")

    # ── 2. Migrate metadata tables ─────────────────────────────────────────────
    logger.info("Step 2 — Migrating metadata tables …")
    total_inserted = 0
    for table in TABLES_IN_ORDER:
        rows = _sqlite_rows(sqlite_url, table)
        if not rows:
            continue
        pk = _PK_OVERRIDES.get(table) or ("id" if "id" in rows[0] else list(rows[0].keys())[0])
        n = await _upsert(engine, table, rows, pk=pk)
        logger.info("  %-35s inserted %d / %d rows", table, n, len(rows))
        total_inserted += n
    logger.info("  Total inserted: %d rows", total_inserted)

    # ── 3. Migrate local files into BYTEA ──────────────────────────────────────
    logger.info("Step 3 — Migrating local audio/video files into PostgreSQL BYTEA …")

    from sqlalchemy import text
    local_storage = Path(settings.STORAGE_LOCAL_PATH)
    multimodal_storage = Path(settings.MULTIMODAL_STORAGE_PATH)

    files_migrated = 0
    files_skipped = 0
    files_missing = 0

    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT id, storage_key, original_filename FROM media_files")
        )
        media_rows = result.fetchall()

    logger.info("  Found %d media_file records to process …", len(media_rows))

    from database.base import async_session_factory
    from src.models import MediaFileData

    for row in media_rows:
        file_id, storage_key, original_filename = row.id, row.storage_key, row.original_filename

        # Skip rows already stored in DB
        if storage_key and storage_key.startswith("db:"):
            files_skipped += 1
            continue

        # Resolve the file on disk
        candidate_paths = [
            local_storage / storage_key,
            multimodal_storage / storage_key,
            Path(storage_key),
        ]
        file_path = next((p for p in candidate_paths if p.exists()), None)

        if file_path is None:
            logger.warning("  File missing for media_file %s (key=%s)", file_id, storage_key)
            files_missing += 1
            continue

        content = file_path.read_bytes()

        async with async_session_factory() as db:
            existing = await db.get(MediaFileData, file_id)
            if existing:
                files_skipped += 1
            else:
                db.add(MediaFileData(file_id=file_id, data=content))
                await db.commit()
                # Update the storage_key to signal DB storage
                await db.execute(
                    text("UPDATE media_files SET storage_key = :key WHERE id = :id"),
                    {"key": f"db:{file_id}", "id": file_id},
                )
                await db.commit()
                files_migrated += 1
                logger.debug("  Migrated %s (%d bytes)", file_id, len(content))

    logger.info(
        "  Files migrated: %d  |  already in DB: %d  |  missing on disk: %d",
        files_migrated, files_skipped, files_missing,
    )

    logger.info("=" * 60)
    logger.info("Migration complete!  PostgreSQL is now the primary store.")
    logger.info("Set STORAGE_PROVIDER=postgres in backend/.env to activate DB file storage.")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
