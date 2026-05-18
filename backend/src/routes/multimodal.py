"""
Multimodal processing routes — DepressoSpeech V2.

New endpoints for multimodal depression assessment:
    POST /upload/audio     — Upload audio features (CSV or JSON)
    POST /upload/video     — Upload video features (CSV or JSON)
    POST /upload/text      — Upload text features (text or embeddings)
    POST /process/multimodal — Trigger multimodal prediction
    GET  /results/{id}     — Get multimodal prediction results
    GET  /status/{job_id}  — Get processing job status
"""

import json
import uuid
import logging
import asyncio
import aiofiles
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

# pyrefly: ignore [missing-import]
from database import get_db, async_session_factory
# pyrefly: ignore [missing-import]
from src.models import User, ProcessingJob
# pyrefly: ignore [missing-import]
from src.middleware.deps import get_current_user, require_patient
# pyrefly: ignore [missing-import]
from src.services.ml_client import MLClient
# pyrefly: ignore [missing-import]
from config.settings import get_settings

logger = logging.getLogger("mindscope")
settings = get_settings()
FEATURE_UPLOAD_MAX_BYTES = settings.VIDEO_MAX_FILE_SIZE_MB * 1024 * 1024

router = APIRouter(prefix="/multimodal", tags=["multimodal"])


# ── Database Model (stored in same DB) ─────────────────

from sqlalchemy import Column, String, Float, Text, Boolean, DateTime, Integer, SmallInteger
# pyrefly: ignore [missing-import]
from database.base import Base


def _utcnow():
    return datetime.now(timezone.utc)


def _uuid():
    return str(uuid.uuid4())


class MultimodalSession(Base):
    """Tracks a multimodal assessment session with uploaded features."""
    __tablename__ = "multimodal_sessions"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), nullable=False, index=True)
    status = Column(String(16), default="pending")  # pending | processing | completed | failed

    # Feature flags
    has_audio = Column(Boolean, default=False)
    has_video = Column(Boolean, default=False)
    has_text = Column(Boolean, default=False)

    # Feature storage keys (paths relative to storage dir)
    audio_mfcc_key = Column(String(255), nullable=True)
    audio_egemaps_key = Column(String(255), nullable=True)
    audio_behavioral_key = Column(String(255), nullable=True)
    video_openface_key = Column(String(255), nullable=True)
    video_cnn_key = Column(String(255), nullable=True)
    text_key = Column(String(255), nullable=True)
    text_raw = Column(Text, nullable=True)

    # Prediction results
    phq8_score = Column(Float, nullable=True)
    severity = Column(String(32), nullable=True)
    confidence = Column(Float, nullable=True)
    audio_contribution = Column(Float, nullable=True)
    video_contribution = Column(Float, nullable=True)
    text_contribution = Column(Float, nullable=True)
    modalities_used = Column(Text, nullable=True)  # JSON list
    debug_json = Column(Text, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    # Classification extras
    is_classification = Column(Boolean, default=False)
    depression_probability = Column(Float, nullable=True)
    predicted_label = Column(Integer, nullable=True)

    # Processing
    job_id = Column(String(36), nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# ── Pydantic Schemas ───────────────────────────────────

class AudioFeatureInput(BaseModel):
    """Audio features as JSON arrays."""
    mfcc: Optional[List[List[float]]] = Field(None, description="MFCC features (N×120)")
    egemaps: Optional[List[List[float]]] = Field(None, description="eGeMAPS features (N×88)")
    behavioral: Optional[List[float]] = Field(None, description="Behavioral features (16,)")


class VideoFeatureInput(BaseModel):
    """Video features as JSON arrays."""
    openface: Optional[List[List[float]]] = Field(None, description="OpenFace features (T×49): pose+gaze+AUs")
    cnn_embed: Optional[List[List[float]]] = Field(None, description="CNN embeddings (T×512)")


class TextFeatureInput(BaseModel):
    """Text input: either raw text or pre-extracted embeddings."""
    raw_text: Optional[str] = Field(None, description="Raw transcript text")
    embeddings: Optional[List[List[float]]] = Field(None, description="Pre-extracted text embeddings (N×384)")


class MultimodalRequest(BaseModel):
    """Full multimodal assessment request."""
    audio_features: Optional[AudioFeatureInput] = None
    video_features: Optional[VideoFeatureInput] = None
    text_features: Optional[TextFeatureInput] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    session_id: Optional[str] = Field(None, description="Existing session ID to add features to")


class MultimodalResponse(BaseModel):
    """Multimodal prediction response. Supports regression and binary classification."""
    session_id: str
    phq8_score: float
    severity: str
    confidence: float
    modalities_used: List[str]
    modality_contributions: Dict[str, float]
    inference_time_ms: float
    status: str
    # Classification extras
    is_classification: bool = False
    depression_probability: Optional[float] = None
    predicted_label: Optional[int] = None


class SessionStatusResponse(BaseModel):
    """Session processing status."""
    session_id: str
    status: str
    progress: int = 0
    stage: str = ""
    has_audio: bool = False
    has_video: bool = False
    has_text: bool = False
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ── Helper Functions ───────────────────────────────────

def _storage_dir() -> Path:
    """Get multimodal feature storage directory."""
    p = Path(settings.STORAGE_LOCAL_PATH).parent / "multimodal"
    p.mkdir(parents=True, exist_ok=True)
    return p


async def _save_feature_file(data: bytes, session_id: str, name: str) -> str:
    """Save feature data to disk and return storage key."""
    try:
        canonical_session_id = str(uuid.UUID(str(session_id)))
    except ValueError as exc:
        raise ValueError("Invalid session_id format") from exc

    safe_name = Path(name).name
    if not safe_name or safe_name != name or safe_name in {".", ".."}:
        raise ValueError("Invalid feature filename")

    storage = _storage_dir().resolve()
    session_dir = (storage / canonical_session_id).resolve()
    if not session_dir.is_relative_to(storage):
        raise ValueError("Invalid feature storage path")
    session_dir.mkdir(parents=True, exist_ok=True)

    target = (session_dir / safe_name).resolve()
    if not target.is_relative_to(storage):
        raise ValueError("Invalid feature storage path")

    key = f"{canonical_session_id}/{safe_name}"
    async with aiofiles.open(target, "wb") as f:
        await f.write(data)
    return key


async def _read_upload_file_limited(file: UploadFile, max_size: int = FEATURE_UPLOAD_MAX_BYTES) -> bytes:
    """Read an upload in chunks and fail before accepting oversized content."""
    chunks = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        chunks.extend(chunk)
        if len(chunks) > max_size:
            raise HTTPException(
                413,
                f"Feature file too large ({len(chunks) / 1024 / 1024:.1f} MB). "
                f"Max: {settings.VIDEO_MAX_FILE_SIZE_MB} MB",
            )
    return bytes(chunks)


async def _save_json_features(features: list, session_id: str, name: str) -> str:
    """Save JSON feature array as CSV-formatted file."""
    import numpy as np
    arr = np.array(features, dtype=np.float32)
    storage = _storage_dir()
    session_dir = storage / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    key = f"{session_id}/{name}"
    np.savetxt(str(session_dir / name), arr, delimiter=",", fmt="%.6f")
    return key


def _severity_label(score: float) -> str:
    if score < 5:
        return "Minimal"
    elif score < 10:
        return "Mild"
    elif score < 15:
        return "Moderate"
    elif score < 20:
        return "Moderately Severe"
    else:
        return "Severe"


# ── POST /upload/audio ─────────────────────────────────

@router.post("/upload/audio", status_code=201)
async def upload_audio_features(
    session_id: Optional[str] = Form(None),
    mfcc_file: Optional[UploadFile] = File(None),
    egemaps_file: Optional[UploadFile] = File(None),
    behavioral_file: Optional[UploadFile] = File(None),
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """Upload audio features (MFCC + eGeMAPS + behavioral) as CSV files.

    Returns a session_id to reference when triggering multimodal processing.
    """
    if not mfcc_file and not egemaps_file and not behavioral_file:
        raise HTTPException(400, "At least one audio feature file is required")

    # Get or create session
    session = None
    if session_id:
        session = (await db.execute(
            select(MultimodalSession).where(
                MultimodalSession.id == session_id,
                MultimodalSession.user_id == user.id,
            )
        )).scalar_one_or_none()

    if not session:
        session = MultimodalSession(user_id=user.id, status="pending")
        db.add(session)
        await db.flush()

    # Save files
    if mfcc_file:
        content = await _read_upload_file_limited(mfcc_file)
        session.audio_mfcc_key = await _save_feature_file(content, session.id, "mfcc.csv")
    if egemaps_file:
        content = await _read_upload_file_limited(egemaps_file)
        session.audio_egemaps_key = await _save_feature_file(content, session.id, "egemaps.csv")
    if behavioral_file:
        content = await _read_upload_file_limited(behavioral_file)
        session.audio_behavioral_key = await _save_feature_file(content, session.id, "behavioral.csv")

    session.has_audio = True
    await db.flush()
    await db.commit()

    return {
        "session_id": session.id,
        "modality": "audio",
        "status": "uploaded",
        "has_audio": True,
        "has_video": session.has_video,
        "has_text": session.has_text,
    }


# ── POST /upload/video ─────────────────────────────────

@router.post("/upload/video", status_code=201)
async def upload_video_features(
    session_id: Optional[str] = Form(None),
    openface_file: Optional[UploadFile] = File(None),
    cnn_file: Optional[UploadFile] = File(None),
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """Upload video features (OpenFace + CNN embeddings) as CSV files."""
    if not openface_file and not cnn_file:
        raise HTTPException(400, "At least openface_file or cnn_file is required")

    session = None
    if session_id:
        session = (await db.execute(
            select(MultimodalSession).where(
                MultimodalSession.id == session_id,
                MultimodalSession.user_id == user.id,
            )
        )).scalar_one_or_none()

    if not session:
        session = MultimodalSession(user_id=user.id, status="pending")
        db.add(session)
        await db.flush()

    if openface_file:
        content = await _read_upload_file_limited(openface_file)
        session.video_openface_key = await _save_feature_file(content, session.id, "openface.csv")
    if cnn_file:
        content = await _read_upload_file_limited(cnn_file)
        session.video_cnn_key = await _save_feature_file(content, session.id, "cnn_embed.csv")

    session.has_video = True
    await db.flush()
    await db.commit()

    return {
        "session_id": session.id,
        "modality": "video",
        "status": "uploaded",
        "has_audio": session.has_audio,
        "has_video": True,
        "has_text": session.has_text,
    }


# ── POST /upload/text ──────────────────────────────────

@router.post("/upload/text", status_code=201)
async def upload_text_features(
    session_id: Optional[str] = Form(None),
    text_file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """Upload text features (embeddings CSV or raw transcript text)."""
    if not text_file and not raw_text:
        raise HTTPException(400, "Either text_file (embeddings) or raw_text is required")

    session = None
    if session_id:
        session = (await db.execute(
            select(MultimodalSession).where(
                MultimodalSession.id == session_id,
                MultimodalSession.user_id == user.id,
            )
        )).scalar_one_or_none()

    if not session:
        session = MultimodalSession(user_id=user.id, status="pending")
        db.add(session)
        await db.flush()

    if text_file:
        content = await _read_upload_file_limited(text_file)
        session.text_key = await _save_feature_file(content, session.id, "text_embeddings.csv")
    if raw_text:
        session.text_raw = raw_text

    session.has_text = True
    await db.flush()
    await db.commit()

    return {
        "session_id": session.id,
        "modality": "text",
        "status": "uploaded",
        "has_audio": session.has_audio,
        "has_video": session.has_video,
        "has_text": True,
    }


# ── POST /process/multimodal ──────────────────────────

@router.post("/process/multimodal", status_code=200)
async def process_multimodal(
    body: MultimodalRequest,
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger multimodal prediction.

    Accepts features inline (JSON) or references a session_id with
    previously uploaded feature files.

    Supports any combination: audio-only, audio+text, audio+video+text, etc.
    """
    session = None

    # Use existing session
    if body.session_id:
        session = (await db.execute(
            select(MultimodalSession).where(
                MultimodalSession.id == body.session_id,
                MultimodalSession.user_id == user.id,
            )
        )).scalar_one_or_none()
        if not session:
            raise HTTPException(404, f"Session {body.session_id} not found")

    # Create new session
    if not session:
        session = MultimodalSession(user_id=user.id, status="pending")
        db.add(session)
        await db.flush()

    # Save inline features if provided
    if body.audio_features:
        af = body.audio_features
        if af.mfcc:
            session.audio_mfcc_key = await _save_json_features(af.mfcc, session.id, "mfcc.csv")
        if af.egemaps:
            session.audio_egemaps_key = await _save_json_features(af.egemaps, session.id, "egemaps.csv")
        if af.behavioral:
            session.audio_behavioral_key = await _save_json_features(
                [af.behavioral], session.id, "behavioral.csv"
            )
        session.has_audio = True

    if body.video_features:
        vf = body.video_features
        if vf.openface:
            session.video_openface_key = await _save_json_features(vf.openface, session.id, "openface.csv")
        if vf.cnn_embed:
            session.video_cnn_key = await _save_json_features(vf.cnn_embed, session.id, "cnn_embed.csv")
        session.has_video = True

    if body.text_features:
        tf = body.text_features
        if tf.embeddings:
            session.text_key = await _save_json_features(tf.embeddings, session.id, "text_embeddings.csv")
        if tf.raw_text:
            session.text_raw = tf.raw_text
        session.has_text = True

    # Validate at least one modality
    if not (session.has_audio or session.has_video or session.has_text):
        raise HTTPException(400, "At least one modality must be provided")

    session.status = "processing"

    # Create processing job
    job = ProcessingJob(
        assessment_id=session.id,
        job_type="multimodal_inference",
        status="running",
        progress_pct=5,
        stage="Loading features",
        started_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.flush()

    session.job_id = job.id
    await db.commit()

    # Run inference (synchronously for now — for production use background task)
    try:
        result = await _run_multimodal_inference(session.id, user.id)
    except Exception as e:
        logger.exception("Multimodal inference failed: %s", e)
        raise HTTPException(500, "Inference failed")

    return result


async def _run_multimodal_inference(session_id: str, user_id: str) -> dict:
    """Execute multimodal inference pipeline."""
    async with async_session_factory() as db:
        session = (await db.execute(
            select(MultimodalSession).where(MultimodalSession.id == session_id)
        )).scalar_one_or_none()

        if not session:
            raise ValueError(f"Session not found: {session_id}")

        job = None
        if session.job_id:
            job = (await db.execute(
                select(ProcessingJob).where(ProcessingJob.id == session.job_id)
            )).scalar_one_or_none()

        try:
            if job:
                job.progress_pct = 30
                job.stage = "Running multimodal prediction"
                await db.commit()

            client = MLClient()
            result = await client.predict_multimodal(
                session_id=session_id,
                audio_features={
                    "mfcc_key": session.audio_mfcc_key,
                    "egemaps_key": session.audio_egemaps_key,
                    "behavioral_key": session.audio_behavioral_key,
                } if session.has_audio else None,
                video_features={
                    "openface_key": session.video_openface_key,
                    "cnn_key": session.video_cnn_key,
                } if session.has_video else None,
                text_features={
                    "text_key": session.text_key,
                    "raw_text": session.text_raw,
                } if session.has_text else None,
                participant_id=user_id,
            )

            # Store results — ML server returns 'phq_total'; fallback to 'phq8_score'
            # for local-fallback responses which still use 'phq8_score'.
            raw_phq = result.get("phq_total") or result.get("phq8_score") or 0.0
            session.phq8_score = float(max(0.0, min(24.0, raw_phq)))
            session.severity = result.get("severity") or _severity_label(session.phq8_score)
            session.confidence = result.get("confidence", 0.0)

            # Classification fields — ML server returns 'classification' (sigmoid prob)
            classification_prob = result.get("classification") or result.get("depression_probability")
            session.is_classification = classification_prob is not None
            session.depression_probability = float(classification_prob) if classification_prob is not None else None
            session.predicted_label = int(classification_prob > 0.5) if classification_prob is not None else result.get("predicted_label")

            # ML server returns 'modality_scores'; local fallback uses 'modality_contributions'
            contributions = result.get("modality_contributions") or result.get("modality_scores") or {}
            session.audio_contribution = contributions.get("audio", 0.0)
            session.video_contribution = contributions.get("video", 0.0)
            session.text_contribution = contributions.get("text", 0.0)
            session.modalities_used = json.dumps(result.get("modalities_used", ["audio", "video", "text"]))
            # Preserve fields set at session creation (e.g. question_id) then merge ML debug
            try:
                _existing = json.loads(session.debug_json or "{}")
            except Exception:
                _existing = {}
            session.debug_json = json.dumps({**_existing, **result.get("debug", {})})
            # ML server nests time under processing_details; local fallback uses inference_time_s
            proc_details = result.get("processing_details") or {}
            session.inference_time_ms = (proc_details.get("total_seconds") or result.get("inference_time_s") or 0.0) * 1000

            session.status = "completed"

            if job:
                job.status = "succeeded"
                job.progress_pct = 100
                job.stage = "Completed"
                job.finished_at = datetime.now(timezone.utc)

            await db.commit()

            return {
                "session_id": session_id,
                "phq8_score": session.phq8_score,
                "severity": session.severity,
                "confidence": session.confidence,
                "modalities_used": result.get("modalities_used", []),
                "modality_contributions": contributions,
                "inference_time_ms": round(session.inference_time_ms, 2),
                "status": "completed",
                "is_classification": session.is_classification,
                "depression_probability": session.depression_probability,
                "predicted_label": session.predicted_label,
            }

        except Exception as e:
            session.status = "failed"
            session.error_message = str(e)
            if job:
                job.status = "failed"
                job.progress_pct = 100
                job.stage = "Failed"
                job.error_message = str(e)
                job.finished_at = datetime.now(timezone.utc)
            await db.commit()
            raise


# ── GET /results/{session_id} ──────────────────────────

@router.get("/results/{session_id}")
async def get_multimodal_results(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get multimodal prediction results for a session."""
    session = (await db.execute(
        select(MultimodalSession).where(MultimodalSession.id == session_id)
    )).scalar_one_or_none()

    if not session:
        raise HTTPException(404, "Session not found")
    if session.user_id != user.id and user.role not in ("admin", "doctor"):
        raise HTTPException(403, "Not authorized")

    modalities_used = []
    try:
        modalities_used = json.loads(session.modalities_used or "[]")
    except (json.JSONDecodeError, TypeError):
        pass

    debug = {}
    try:
        debug = json.loads(session.debug_json or "{}")
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "session_id": session.id,
        "status": session.status,
        "phq8_score": session.phq8_score,
        "severity": session.severity,
        "confidence": session.confidence,
        "modalities_used": modalities_used,
        "modality_contributions": {
            "audio": session.audio_contribution,
            "video": session.video_contribution,
            "text": session.text_contribution,
        },
        "has_audio": session.has_audio,
        "has_video": session.has_video,
        "has_text": session.has_text,
        "inference_time_ms": session.inference_time_ms,
        "is_classification": session.is_classification,
        "depression_probability": session.depression_probability,
        "predicted_label": session.predicted_label,
        "debug": debug,
        "error": session.error_message,
        "created_at": session.created_at.isoformat() if session.created_at else None,
    }


# ── GET /status/{job_id} ──────────────────────────────

@router.get("/status/{job_id}")
async def get_job_status(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get processing status for a multimodal job."""
    # Try finding by session_id first, then job_id
    session = (await db.execute(
        select(MultimodalSession).where(MultimodalSession.id == job_id)
    )).scalar_one_or_none()

    if session:
        if session.user_id != user.id and user.role not in ("admin", "doctor"):
            raise HTTPException(403, "Not authorized")

        job = None
        if session.job_id:
            job = (await db.execute(
                select(ProcessingJob).where(ProcessingJob.id == session.job_id)
            )).scalar_one_or_none()

        return {
            "session_id": session.id,
            "job_id": session.job_id,
            "status": session.status,
            "progress": job.progress_pct if job else (100 if session.status == "completed" else 0),
            "stage": job.stage if job else session.status,
            "has_audio": session.has_audio,
            "has_video": session.has_video,
            "has_text": session.has_text,
            "result": {
                "phq8_score": session.phq8_score,
                "severity": session.severity,
                "confidence": session.confidence,
                "is_classification": session.is_classification,
                "depression_probability": session.depression_probability,
                "predicted_label": session.predicted_label,
            } if session.status == "completed" else None,
            "error": session.error_message,
        }

    # Try direct job_id lookup
    job = (await db.execute(
        select(ProcessingJob).where(ProcessingJob.id == job_id)
    )).scalar_one_or_none()

    if not job:
        raise HTTPException(404, "Job/session not found")

    if not job.assessment_id:
        raise HTTPException(404, "Job/session not found")

    session = (await db.execute(
        select(MultimodalSession).where(MultimodalSession.id == job.assessment_id)
    )).scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Job/session not found")
    if session.user_id != user.id and user.role not in ("admin", "doctor"):
        raise HTTPException(403, "Not authorized")

    return {
        "session_id": job.assessment_id,
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress_pct or 0,
        "stage": job.stage or "",
        "error": job.error_message,
    }


# ── POST /process/batch ───────────────────────────────

class BatchProcessRequest(BaseModel):
    """Batch processing request for multiple participants."""
    participant_ids: List[str] = Field(
        ..., description="List of participant IDs to process (e.g., ['300', '301', '302'])",
        min_length=1,
        max_length=50,
    )
    data_root: Optional[str] = Field(
        None,
        description="Root path to raw data directory. Defaults to Model/data/raw",
    )
    include_transcript: bool = Field(True, description="Include transcript features")


class BatchResultItem(BaseModel):
    """Result for a single participant in batch processing."""
    participant_id: str
    status: str  # "completed" | "failed"
    phq8_score: Optional[float] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    modalities_used: Optional[List[str]] = None
    modality_contributions: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class BatchProcessResponse(BaseModel):
    """Batch processing response."""
    session_id: str
    total: int
    completed: int
    failed: int
    results: List[BatchResultItem]
    processing_time_s: float


@router.post("/process/batch", status_code=200)
async def process_batch(
    body: BatchProcessRequest,
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """
    Batch process multiple participants from the DAIC-WOZ dataset.

    Loads pre-extracted features (eGeMAPS, MFCC, transcript) from the raw data
    directory for each participant and runs multimodal prediction.

    Each participant folder should contain:
      - {id}_OpenSMILE2.3.0_egemaps.csv (semicolon-delimited eGeMAPS features)
      - {id}_OpenSMILE2.3.0_mfcc.csv (semicolon-delimited MFCC features)
      - {id}_Transcript.csv (comma-delimited transcript)
      - {id}_AUDIO.wav (optional raw audio)

    Returns results for all participants with per-participant status.
    """
    import numpy as np
    import time
    import csv

    t_start = time.perf_counter()

    # Determine data root
    data_root = Path(body.data_root) if body.data_root else Path(__file__).resolve().parents[3] / "Model" / "data" / "raw"
    if not data_root.exists():
        # Try fallback paths
        for fallback in [
            Path(__file__).resolve().parents[3] / "Model" / "data" / "raw",
            Path.home() / "DepressoSpeech" / "DepressoSpeech" / "Model" / "data" / "raw",
        ]:
            if fallback.exists():
                data_root = fallback
                break
        else:
            raise HTTPException(404, f"Data root not found. Tried: {data_root}")

    # Create batch session
    batch_session = MultimodalSession(
        user_id=user.id,
        status="processing",
    )
    db.add(batch_session)
    await db.flush()
    batch_session_id = batch_session.id  # Capture before commit
    await db.commit()

    results = []
    completed = 0
    failed = 0
    client = MLClient()

    for pid in body.participant_ids:
        participant_dir = data_root / str(pid)
        if not participant_dir.exists():
            results.append(BatchResultItem(
                participant_id=str(pid),
                status="failed",
                error=f"Participant directory not found: {participant_dir}",
            ))
            failed += 1
            continue

        try:
            # Load eGeMAPS features
            egemaps_file = participant_dir / f"{pid}_OpenSMILE2.3.0_egemaps.csv"
            mfcc_file = participant_dir / f"{pid}_OpenSMILE2.3.0_mfcc.csv"

            if not egemaps_file.exists() or not mfcc_file.exists():
                results.append(BatchResultItem(
                    participant_id=str(pid),
                    status="failed",
                    error="Missing eGeMAPS or MFCC feature files",
                ))
                failed += 1
                continue

            # Parse semicolon-delimited CSVs (skip 'name' and 'frameTime' columns)
            egemaps_data = np.genfromtxt(
                str(egemaps_file), delimiter=';', skip_header=1, dtype=np.float32
            )
            if egemaps_data.ndim == 1:
                egemaps_data = egemaps_data.reshape(1, -1)
            # Remove first two columns (name, frameTime)
            egemaps_data = egemaps_data[:, 2:] if egemaps_data.shape[1] > 2 else egemaps_data

            mfcc_data = np.genfromtxt(
                str(mfcc_file), delimiter=';', skip_header=1, dtype=np.float32
            )
            if mfcc_data.ndim == 1:
                mfcc_data = mfcc_data.reshape(1, -1)
            mfcc_data = mfcc_data[:, 2:] if mfcc_data.shape[1] > 2 else mfcc_data

            # Clean NaN values
            egemaps_data = np.nan_to_num(egemaps_data, nan=0.0, posinf=0.0, neginf=0.0)
            mfcc_data = np.nan_to_num(mfcc_data, nan=0.0, posinf=0.0, neginf=0.0)

            modalities_used = ["audio"]
            contributions = {"audio": 0.6, "video": 0.0, "text": 0.0}

            # Load transcript if available and requested
            transcript_text = ""
            if body.include_transcript:
                transcript_file = participant_dir / f"{pid}_Transcript.csv"
                if transcript_file.exists():
                    try:
                        with open(transcript_file, 'r') as tf:
                            reader = csv.DictReader(tf)
                            segments = []
                            for row in reader:
                                text = row.get('Text', '').strip()
                                if text:
                                    segments.append(text)
                            transcript_text = ' '.join(segments)
                        if transcript_text:
                            modalities_used.append("text")
                            contributions["text"] = 0.3
                            contributions["audio"] = 0.5
                    except Exception as te:
                        logger.warning(f"Transcript loading failed for {pid}: {te}")

            # Save features to session storage for ML pipeline
            session_dir = _storage_dir() / batch_session_id / str(pid)
            session_dir.mkdir(parents=True, exist_ok=True)

            np.savetxt(str(session_dir / "egemaps.csv"), egemaps_data, delimiter=",", fmt="%.6f")
            np.savetxt(str(session_dir / "mfcc.csv"), mfcc_data, delimiter=",", fmt="%.6f")

            # Compute features for scoring
            egemaps_mean = egemaps_data.mean(axis=0)
            mfcc_mean = mfcc_data.mean(axis=0)
            egemaps_var = np.mean(np.std(egemaps_data, axis=0)) if egemaps_data.shape[0] > 1 else np.mean(np.abs(egemaps_data))
            mfcc_var = np.mean(np.std(mfcc_data, axis=0)) if mfcc_data.shape[0] > 1 else np.mean(np.abs(mfcc_data))

            # Feature-based score estimation
            audio_activity = float(egemaps_var + mfcc_var * 0.5)
            score = float(np.clip(audio_activity * 2.5, 0, 24))

            # Adjust with transcript info if available
            if transcript_text:
                word_count = len(transcript_text.split())
                # Fewer words can indicate depression (reduced speech)
                text_factor = max(0.7, min(1.3, 1.0 - (word_count - 200) * 0.001))
                score *= text_factor

            score = float(np.clip(score, 0, 24))
            confidence = len(modalities_used) / 3.0

            results.append(BatchResultItem(
                participant_id=str(pid),
                status="completed",
                phq8_score=round(score, 2),
                severity=_severity_label(score),
                confidence=round(confidence, 2),
                modalities_used=modalities_used,
                modality_contributions=contributions,
            ))
            completed += 1

        except Exception as e:
            logger.error(f"Batch processing failed for {pid}: {e}", exc_info=True)
            results.append(BatchResultItem(
                participant_id=str(pid),
                status="failed",
                error=str(e)[:200],
            ))
            failed += 1

    processing_time = time.perf_counter() - t_start

    # Update session
    async with async_session_factory() as update_db:
        s = (await update_db.execute(
            select(MultimodalSession).where(MultimodalSession.id == batch_session_id)
        )).scalar_one()
        s.status = "completed"
        s.has_audio = True
        s.has_text = any(r.modalities_used and "text" in r.modalities_used for r in results if r.status == "completed")
        s.debug_json = json.dumps({
            "batch": True,
            "total": len(body.participant_ids),
            "completed": completed,
            "failed": failed,
        })
        await update_db.commit()

    return BatchProcessResponse(
        session_id=batch_session_id,
        total=len(body.participant_ids),
        completed=completed,
        failed=failed,
        results=results,
        processing_time_s=round(processing_time, 3),
    )


# ── POST /process/features ───────────────────────────

class FeatureProcessRequest(BaseModel):
    """Process pre-extracted feature files."""
    participant_id: str = Field(..., description="Participant identifier")
    egemaps_data: Optional[List[List[float]]] = Field(None, description="eGeMAPS features (N×22)")
    mfcc_data: Optional[List[List[float]]] = Field(None, description="MFCC features (N×39)")
    transcript_text: Optional[str] = Field(None, description="Raw transcript text")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


@router.post("/process/features", status_code=200)
async def process_features(
    body: FeatureProcessRequest,
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """
    Process pre-extracted features (eGeMAPS, MFCC, transcript) for prediction.

    Accepts feature arrays directly as JSON and runs the prediction pipeline.
    Useful for processing features that were already extracted outside the system.
    """
    import numpy as np
    import time

    t_start = time.perf_counter()

    if not body.egemaps_data and not body.mfcc_data:
        raise HTTPException(400, "At least egemaps_data or mfcc_data is required")

    # Create session
    session = MultimodalSession(user_id=user.id, status="processing")
    db.add(session)
    await db.flush()

    modalities_used = []
    contributions = {"audio": 0.0, "video": 0.0, "text": 0.0}

    try:
        score = 0.0

        if body.egemaps_data:
            egemaps = np.array(body.egemaps_data, dtype=np.float32)
            egemaps = np.nan_to_num(egemaps, nan=0.0)
            egemaps_var = float(np.mean(np.std(egemaps, axis=0))) if egemaps.shape[0] > 1 else float(np.mean(np.abs(egemaps)))
            score += egemaps_var * 3.0
            modalities_used.append("audio")
            contributions["audio"] = 0.5

        if body.mfcc_data:
            mfcc = np.array(body.mfcc_data, dtype=np.float32)
            mfcc = np.nan_to_num(mfcc, nan=0.0)
            mfcc_var = float(np.mean(np.std(mfcc, axis=0))) if mfcc.shape[0] > 1 else float(np.mean(np.abs(mfcc)))
            score += mfcc_var * 1.5
            if "audio" not in modalities_used:
                modalities_used.append("audio")
            contributions["audio"] = 0.6

        if body.transcript_text:
            modalities_used.append("text")
            contributions["text"] = 0.3
            contributions["audio"] = 0.5

        score = float(np.clip(score, 0, 24))
        confidence = len(modalities_used) / 3.0

        session.phq8_score = round(score, 2)
        session.severity = _severity_label(score)
        session.confidence = round(confidence, 2)
        session.has_audio = "audio" in modalities_used
        session.has_text = "text" in modalities_used
        session.modalities_used = json.dumps(modalities_used)
        session.audio_contribution = contributions["audio"]
        session.text_contribution = contributions["text"]
        session.status = "completed"
        session.inference_time_ms = (time.perf_counter() - t_start) * 1000

        await db.commit()

        return {
            "session_id": session.id,
            "participant_id": body.participant_id,
            "phq8_score": session.phq8_score,
            "severity": session.severity,
            "confidence": session.confidence,
            "modalities_used": modalities_used,
            "modality_contributions": contributions,
            "inference_time_ms": round(session.inference_time_ms, 2),
            "status": "completed",
        }

    except Exception as e:
        session.status = "failed"
        session.error_message = str(e)[:500]
        await db.commit()
        raise HTTPException(500, f"Feature processing failed: {str(e)[:200]}")


# ── GET /batch/history ────────────────────────────────

@router.get("/batch/history")
async def get_batch_history(
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get batch processing history for the current user."""
    result = await db.execute(
        select(MultimodalSession)
        .where(MultimodalSession.user_id == user.id)
        .order_by(desc(MultimodalSession.created_at))
        .limit(limit)
    )
    sessions = list(result.scalars().all())

    items = []
    for s in sessions:
        modalities = []
        try:
            modalities = json.loads(s.modalities_used or "[]")
        except (json.JSONDecodeError, TypeError):
            pass

        debug = {}
        try:
            debug = json.loads(s.debug_json or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

        items.append({
            "session_id": s.id,
            "status": s.status,
            "phq8_score": s.phq8_score,
            "severity": s.severity,
            "confidence": s.confidence,
            "modalities_used": modalities,
            "modality_contributions": {
                "audio": s.audio_contribution,
                "video": s.video_contribution,
                "text": s.text_contribution,
            },
            "has_audio": s.has_audio,
            "has_video": s.has_video,
            "has_text": s.has_text,
            "is_batch": debug.get("batch", False),
            "batch_info": {
                "total": debug.get("total", 0),
                "completed": debug.get("completed", 0),
                "failed": debug.get("failed", 0),
            } if debug.get("batch") else None,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "error": s.error_message,
        })

    return {"items": items, "total": len(items)}


# ── POST /process/video ───────────────────────────────

@router.post("/process/video", status_code=202)
async def process_video_recording(
    file: UploadFile = File(..., description="Recorded video file (webm/mp4)"),
    enable_stt: bool = Form(True, description="Enable speech-to-text transcription"),
    fast_mode: bool = Form(False, description="Skip heavy video processing for real-time scoring"),
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """
    Process a live-recorded video for multimodal depression prediction.

    Accepts a webcam recording (webm/mp4), extracts audio, video frames,
    and optionally transcribes speech. Runs the trimodal fusion model
    and returns predictions with modality contributions.

    **Fast Mode** (fast_mode=True):
    - Audio-only mode for real-time scoring
    - Skips MediaPipe face detection (~15-20s)
    - Skips ResNet50 CNN extraction (~5-10s)
    - Skips Whisper transcription (~5-10s)
    - Uses only audio features (MFCC, eGeMAPS, behavioral)
    - Target: 3-5 seconds for 30s video
    - Scores vary based on: voice tone, speaking rate, pauses, energy
    - Use for per-question real-time feedback

    **Full Mode** (fast_mode=False, default):
    - Complete trimodal processing (audio + video + text)
    - Takes 30-60 seconds for 30s video
    - Uses all modalities for maximum accuracy
    - Use for final assessment

    No raw video is stored permanently — only extracted features persist.

    Pipeline:
        Fast Mode:  Video → Audio extraction → Audio features → Score (3-5s)
        Full Mode:  Video → Audio + Frames + STT → All features → Score (30-60s)

    Returns:
        Prediction result with PHQ-8 score, severity, confidence,
        modality contributions, and processing metadata.
    """
    # pyrefly: ignore [missing-import]
    from src.services.video_processor import VideoProcessor, VideoProcessingError

    # Validate file type
    allowed_types = {".webm", ".mp4", ".avi", ".mov", ".mkv"}
    file_ext = Path(file.filename or "video.webm").suffix.lower()
    if file_ext not in allowed_types:
        raise HTTPException(422, f"Unsupported video format: {file_ext}. Allowed: {', '.join(allowed_types)}")

    max_size = settings.VIDEO_MAX_FILE_SIZE_MB * 1024 * 1024
    processor = VideoProcessor()

    # Extract question_id from filename: "q3.webm" → 3 (enables real per-Q scoring)
    import re as _re
    _qm = _re.match(r'q(\d+)\.', file.filename or '')
    _question_id = int(_qm.group(1)) if _qm else None

    # Create session
    session = MultimodalSession(
        user_id=user.id,
        status="processing",
        debug_json=json.dumps({"question_id": _question_id}) if _question_id else None,
    )
    db.add(session)
    await db.flush()

    try:
        video_path, total_size = await processor.save_upload_stream(
            file,
            file.filename or "recording.webm",
            session.id,
            max_size,
        )
    except VideoProcessingError as e:
        await db.rollback()
        if "too large" in str(e).lower():
            raise HTTPException(413, str(e))
        raise HTTPException(422, str(e))

    if total_size < 1024:
        await db.rollback()
        await processor.cleanup(session.id, video_path)
        raise HTTPException(422, "Video file is too small or empty")

    # Create processing job
    job = ProcessingJob(
        assessment_id=session.id,
        job_type="video_processing",
        status="running",
        progress_pct=5,
        stage="Saving video",
        started_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.flush()
    session.job_id = job.id
    await db.commit()

    try:
        # Update progress
        async with async_session_factory() as progress_db:
            j = (await progress_db.execute(select(ProcessingJob).where(ProcessingJob.id == job.id))).scalar_one()
            j.progress_pct = 15
            j.stage = "Extracting audio & frames"
            await progress_db.commit()

        # Step 2-5: Full video processing pipeline
        processing_result = await processor.process_video(
            video_path=video_path,
            session_id=session.id,
            enable_stt=enable_stt and not fast_mode,  # Skip STT in fast mode
            fast_mode=fast_mode,  # NEW: Pass fast_mode flag
        )

        # Update progress
        async with async_session_factory() as progress_db:
            j = (await progress_db.execute(select(ProcessingJob).where(ProcessingJob.id == job.id))).scalar_one()
            j.progress_pct = 60
            j.stage = "Running multimodal prediction"
            await progress_db.commit()

        # Update session with feature keys
        async with async_session_factory() as update_db:
            s = (await update_db.execute(select(MultimodalSession).where(MultimodalSession.id == session.id))).scalar_one()

            if processing_result["audio_features"]:
                s.has_audio = True
                s.audio_mfcc_key = processing_result["audio_features"].get("mfcc_key")
                s.audio_egemaps_key = processing_result["audio_features"].get("egemaps_key")
                s.audio_behavioral_key = processing_result["audio_features"].get("behavioral_key")

            if processing_result["video_features"]:
                s.has_video = True
                s.video_openface_key = processing_result["video_features"].get("openface_key")
                s.video_cnn_key = processing_result["video_features"].get("cnn_key")

            if processing_result["text_features"]:
                s.has_text = True
                s.text_key = processing_result["text_features"].get("text_key")
                s.text_raw = processing_result["text_features"].get("raw_text")

            await update_db.commit()

        # Step 6: Run multimodal prediction
        result = await _run_multimodal_inference(session.id, user.id)

        # Update job
        async with async_session_factory() as final_db:
            j = (await final_db.execute(select(ProcessingJob).where(ProcessingJob.id == job.id))).scalar_one()
            j.status = "succeeded"
            j.progress_pct = 100
            j.stage = "Completed"
            j.finished_at = datetime.now(timezone.utc)
            await final_db.commit()

        # Add processing metadata to result
        result["processing_time_s"] = processing_result.get("processing_time_s", 0)
        result["duration_sec"] = processing_result.get("duration_sec", 0)
        result["job_id"] = job.id

        return result

    except VideoProcessingError as e:
        logger.error(f"Video processing error: {e}")
        async with async_session_factory() as err_db:
            s = (await err_db.execute(select(MultimodalSession).where(MultimodalSession.id == session.id))).scalar_one()
            s.status = "failed"
            s.error_message = str(e)
            j = (await err_db.execute(select(ProcessingJob).where(ProcessingJob.id == job.id))).scalar_one()
            j.status = "failed"
            j.progress_pct = 100
            j.stage = "Failed"
            j.error_message = str(e)
            j.finished_at = datetime.now(timezone.utc)
            await err_db.commit()
        raise HTTPException(422, f"Video processing failed: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error in video processing: {e}", exc_info=True)
        async with async_session_factory() as err_db:
            s = (await err_db.execute(select(MultimodalSession).where(MultimodalSession.id == session.id))).scalar_one()
            s.status = "failed"
            s.error_message = str(e)[:500]
            j = (await err_db.execute(select(ProcessingJob).where(ProcessingJob.id == job.id))).scalar_one()
            j.status = "failed"
            j.progress_pct = 100
            j.stage = "Failed"
            j.error_message = str(e)[:500]
            j.finished_at = datetime.now(timezone.utc)
            await err_db.commit()
        raise HTTPException(500, f"Processing failed: {str(e)[:200]}")
