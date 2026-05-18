"""Audio file upload routes."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from src.models import MediaFile, User
from src.middleware.deps import get_current_user, require_patient
from src.services.storage_service import get_storage_service
from config.settings import get_settings

router = APIRouter(prefix="/files", tags=["files"])
settings = get_settings()
logger = logging.getLogger("mindscope")

# Accept both audio and video webm blobs (video recorder sends video/webm)
_VIDEO_EXTS = {e.strip() for e in settings.VIDEO_ALLOWED_EXTENSIONS.split(",")}
ALLOWED_EXTENSIONS = settings.allowed_extensions_set | _VIDEO_EXTS
MAX_SIZE_BYTES = max(
    settings.AUDIO_MAX_FILE_SIZE_MB,
    settings.VIDEO_MAX_FILE_SIZE_MB,
) * 1024 * 1024


@router.post("/audio/upload", status_code=201)
async def upload_audio(
    file: UploadFile = File(...),
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """Direct multipart audio upload. Returns a fileId for use in assessment answers."""
    # Validate extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )
    else:
        ext = ".webm"

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {settings.AUDIO_MAX_FILE_SIZE_MB} MB",
        )

    # Create the media_files row FIRST and commit it so the FK in
    # media_file_data is satisfied when PostgreSQLStorageService.save()
    # runs (it opens a separate DB session and cannot see uncommitted rows).
    file_id = str(uuid.uuid4())
    media = MediaFile(
        id=file_id,
        owner_user_id=user.id,
        original_filename=file.filename,
        storage_key="",          # filled in after storage.save()
        mime_type=file.content_type,
        file_size=len(content),
        status="pending",
    )
    db.add(media)
    await db.commit()            # media_files row now visible to all sessions

    # Persist the binary — FK constraint is now satisfied
    try:
        storage = get_storage_service()
        storage_key = await storage.save(content, file_id, ext)
    except Exception as exc:
        logger.error("Storage save failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Could not save file: {exc}") from exc

    # Update with the real storage key and mark available
    media.storage_key = storage_key
    media.status = "available"
    await db.commit()

    return {
        "fileId": media.id,
        "status": "available",
        "fileName": file.filename,
        "size": len(content),
    }


@router.get("/audio/{file_id}")
async def get_audio_file(
    file_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stream a stored audio file for authorized report playback."""
    media = (await db.execute(
        select(MediaFile).where(MediaFile.id == file_id)
    )).scalar_one_or_none()

    if not media:
        raise HTTPException(status_code=404, detail="Audio file not found")
    if media.owner_user_id != user.id and user.role not in ("admin", "doctor"):
        raise HTTPException(status_code=403, detail="Not authorized")

    storage = get_storage_service()
    return await storage.serve(
        media.storage_key,
        media_type=media.mime_type or "audio/webm",
        filename=media.original_filename or media.storage_key,
    )
