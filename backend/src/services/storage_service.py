"""Pluggable file-storage backend.

Providers
---------
local     (default)  – stores files on the local filesystem.
postgres             – stores raw bytes as BYTEA in the media_file_data table.
                       No external storage needed; works out-of-the-box with PG.
s3                   – stores files in an AWS S3 (or compatible) bucket.
                       Requires: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                                 S3_BUCKET_NAME in the environment / .env file.

The active provider is chosen by the STORAGE_PROVIDER env-var.
The singleton is created once and reused for the life of the process.
"""

from __future__ import annotations

import asyncio
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse

logger = logging.getLogger("mindscope.storage")


# ── Abstract base ──────────────────────────────────────────────────────────────

class StorageService(ABC):
    """Common interface for reading and writing media files."""

    @abstractmethod
    async def save(self, content: bytes, file_id: str, ext: str) -> str:
        """Persist *content* and return the opaque *storage_key*."""

    @abstractmethod
    async def serve(self, storage_key: str, media_type: str, filename: str):
        """Return a FastAPI response that streams / redirects the file."""

    @abstractmethod
    async def delete(self, storage_key: str) -> None:
        """Remove a previously saved file (best-effort; no error on missing)."""

    @abstractmethod
    async def read_bytes(self, storage_key: str) -> bytes:
        """Return the raw bytes of a stored file (raises if missing)."""

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """Return True if the file can be served."""


# ── Local filesystem ───────────────────────────────────────────────────────────

class LocalStorageService(StorageService):
    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        logger.info("LocalStorageService: base_dir=%s", self._base)

    def _path(self, storage_key: str) -> Path:
        return self._base / storage_key

    async def save(self, content: bytes, file_id: str, ext: str) -> str:
        storage_key = f"{file_id}{ext}"
        async with aiofiles.open(self._path(storage_key), "wb") as fh:
            await fh.write(content)
        logger.debug("LocalStorage saved %s (%d bytes)", storage_key, len(content))
        return storage_key

    async def serve(self, storage_key: str, media_type: str, filename: str):
        file_path = self._path(storage_key)
        if not file_path.exists():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="File missing from storage")
        return FileResponse(str(file_path), media_type=media_type, filename=filename)

    async def delete(self, storage_key: str) -> None:
        try:
            self._path(storage_key).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("LocalStorage delete failed for %s: %s", storage_key, exc)

    async def read_bytes(self, storage_key: str) -> bytes:
        file_path = self._path(storage_key)
        if not file_path.exists():
            raise FileNotFoundError(f"Local file not found: {file_path}")
        async with aiofiles.open(file_path, "rb") as fh:
            return await fh.read()

    def exists(self, storage_key: str) -> bool:
        return self._path(storage_key).exists()


# ── AWS S3 (or S3-compatible) ─────────────────────────────────────────────────

class S3StorageService(StorageService):
    """Stores files in an S3 bucket.

    Serving uses signed GET URLs (1-hour expiry) – the client is redirected.
    This offloads bandwidth from the API server and works on all hosting
    platforms (Render, Railway, Heroku, etc.).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "audio/",
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
        endpoint_url: str | None = None,
        url_expiry_seconds: int = 3600,
    ) -> None:
        import boto3
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._expiry = url_expiry_seconds
        self._loop = None
        session = boto3.session.Session()
        self._s3 = session.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
            endpoint_url=endpoint_url or None,
        )
        logger.info(
            "S3StorageService: bucket=%s prefix=%s region=%s", bucket, prefix, region
        )

    def _key(self, storage_key: str) -> str:
        if storage_key.startswith(self._prefix):
            return storage_key
        return f"{self._prefix}{storage_key}"

    async def save(self, content: bytes, file_id: str, ext: str) -> str:
        storage_key = f"{file_id}{ext}"
        s3_key = self._key(storage_key)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._s3.put_object(Bucket=self._bucket, Key=s3_key, Body=content),
        )
        logger.debug("S3Storage saved %s (%d bytes)", s3_key, len(content))
        return storage_key

    async def serve(self, storage_key: str, media_type: str, filename: str):
        s3_key = self._key(storage_key)
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            None,
            lambda: self._s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self._bucket,
                    "Key": s3_key,
                    "ResponseContentDisposition": f'inline; filename="{filename}"',
                    "ResponseContentType": media_type,
                },
                ExpiresIn=self._expiry,
            ),
        )
        return RedirectResponse(url=url, status_code=307)

    async def delete(self, storage_key: str) -> None:
        s3_key = self._key(storage_key)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._s3.delete_object(Bucket=self._bucket, Key=s3_key),
            )
        except Exception as exc:
            logger.warning("S3Storage delete failed for %s: %s", s3_key, exc)

    async def read_bytes(self, storage_key: str) -> bytes:
        s3_key = self._key(storage_key)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._s3.get_object(Bucket=self._bucket, Key=s3_key),
        )
        return response["Body"].read()

    def exists(self, storage_key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._key(storage_key))
            return True
        except Exception:
            return False


# ── PostgreSQL BYTEA storage ───────────────────────────────────────────────────

class PostgreSQLStorageService(StorageService):
    """Stores raw file bytes directly in the PostgreSQL database (BYTEA).

    storage_key convention:  "db:<file_id>"
    The binary payload lives in media_file_data.data (one row per file).

    Advantages
    ----------
    - Zero external dependencies — everything in one DB
    - Works on any hosted PostgreSQL (Railway, Supabase, Neon, RDS, etc.)
    - Transactional: upload + metadata record are committed together

    Limitations
    -----------
    - Not suitable for very large files (>100 MB) in a low-RAM environment
    - Backups include binary data — DB dumps become large
    """

    DB_PREFIX = "db:"

    def __init__(self) -> None:
        logger.info("PostgreSQLStorageService initialised — files stored in BYTEA")

    @staticmethod
    def _file_id_from_key(storage_key: str) -> str:
        return storage_key.removeprefix(PostgreSQLStorageService.DB_PREFIX)

    async def save(self, content: bytes, file_id: str, ext: str) -> str:
        """Insert binary content into media_file_data and return storage_key."""
        from database.base import async_session_factory
        from src.models import MediaFileData

        storage_key = f"{self.DB_PREFIX}{file_id}"
        async with async_session_factory() as db:
            existing = await db.get(MediaFileData, file_id)
            if existing:
                existing.data = content
            else:
                db.add(MediaFileData(file_id=file_id, data=content))
            await db.commit()
        logger.debug("PGStorage saved %s (%d bytes)", storage_key, len(content))
        return storage_key

    async def serve(self, storage_key: str, media_type: str, filename: str):
        """Read bytes from DB and stream back as an HTTP response."""
        from database.base import async_session_factory
        from src.models import MediaFileData

        fid = self._file_id_from_key(storage_key)
        async with async_session_factory() as db:
            row = await db.get(MediaFileData, fid)
        if row is None:
            raise HTTPException(status_code=404, detail="File not found in database")
        return Response(
            content=row.data,
            media_type=media_type,
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    async def delete(self, storage_key: str) -> None:
        from database.base import async_session_factory
        from src.models import MediaFileData

        fid = self._file_id_from_key(storage_key)
        async with async_session_factory() as db:
            row = await db.get(MediaFileData, fid)
            if row:
                await db.delete(row)
                await db.commit()

    async def read_bytes(self, storage_key: str) -> bytes:
        """Read raw bytes from the database BYTEA column."""
        from database.base import async_session_factory
        from src.models import MediaFileData

        fid = self._file_id_from_key(storage_key)
        async with async_session_factory() as db:
            row = await db.get(MediaFileData, fid)
        if row is None:
            raise FileNotFoundError(f"File not found in database: {storage_key}")
        return bytes(row.data)

    def exists(self, storage_key: str) -> bool:
        """Synchronous existence check — not usable for async DB; always True for DB keys."""
        return storage_key.startswith(self.DB_PREFIX)


# ── Singleton factory ──────────────────────────────────────────────────────────

_instance: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Return the process-level singleton StorageService.

    Thread-safe for the common FastAPI/asyncio single-process deployment.
    """
    global _instance
    if _instance is not None:
        return _instance

    from config.settings import get_settings
    s = get_settings()
    provider = getattr(s, "STORAGE_PROVIDER", "local").lower()

    if provider == "postgres":
        _instance = PostgreSQLStorageService()
    elif provider == "s3":
        _instance = S3StorageService(
            bucket=getattr(s, "S3_BUCKET_NAME", ""),
            prefix=getattr(s, "S3_KEY_PREFIX", "audio/"),
            region=getattr(s, "AWS_REGION", "us-east-1"),
            access_key=getattr(s, "AWS_ACCESS_KEY_ID", None) or None,
            secret_key=getattr(s, "AWS_SECRET_ACCESS_KEY", None) or None,
            endpoint_url=getattr(s, "S3_ENDPOINT_URL", None) or None,
            url_expiry_seconds=getattr(s, "S3_URL_EXPIRY_SECONDS", 3600),
        )
    else:
        _instance = LocalStorageService(s.STORAGE_LOCAL_PATH)

    return _instance
