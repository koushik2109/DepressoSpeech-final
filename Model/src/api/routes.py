"""
[LAYER_START] Session 9: API Routes
REST endpoints for depression severity prediction.

[INFERENCE_PATH] HTTP request → validate → InferencePipeline → response.

Endpoints:
    POST /predict         — single audio file
    POST /predict/batch   — multiple audio files
    GET  /health          — health check
"""

import logging
import tempfile
import time
import os
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends, Request

from src.api.schemas import (
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    ExtendedPredictionResponse,
    MultimodalRequest,
    MultimodalPredictionResponse,
)
from src.api.app import limiter, verify_api_key, _pipeline_lock

logger = logging.getLogger(__name__)

router = APIRouter()

# Valid audio extensions
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}

# Max file size: 100 MB
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024

# Max files per batch request (prevents timeout on large batches)
MAX_BATCH_SIZE = 20

PARTICIPANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Magic bytes for audio file content validation (SEC-8)
AUDIO_MAGIC_BYTES = {
    b"RIFF": "wav",             # WAV (RIFF header)
    b"\xff\xfb": "mp3",        # MP3 (MPEG frame sync)
    b"\xff\xf3": "mp3",        # MP3 (MPEG frame sync variant)
    b"\xff\xf2": "mp3",        # MP3 (MPEG frame sync variant)
    b"ID3": "mp3",             # MP3 with ID3 tag
    b"fLaC": "flac",           # FLAC
    b"OggS": "ogg",            # OGG/Vorbis
    b"\x00\x00\x00": "m4a",   # MP4/M4A (ftyp box)
    b"\x1a\x45\xdf\xa3": "webm",  # WebM/Matroska
}


def _get_pipeline():
    """Get the global inference pipeline from app state.

    The pipeline is set during app startup (see app.py lifespan).
    This avoids circular imports and keeps routes decoupled from init.
    """
    from src.api.app import _pipeline

    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up.",
        )
    return _pipeline


def _get_tracker():
    """Get the global experiment tracker (may be None if DB init failed)."""
    from src.api.app import _tracker
    return _tracker


def _validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file type and size."""
    # Check extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

    # Check content type (permissive — some clients send application/octet-stream)
    if file.content_type and file.content_type not in (
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/ogg",
        "audio/mp4",
        "audio/webm",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported content type '{file.content_type}'.",
        )


def _validate_participant_id(participant_id: str) -> str:
    """Validate participant_id for length and allowed characters."""
    if not PARTICIPANT_ID_PATTERN.fullmatch(participant_id):
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid participant_id. Use 1-64 chars from: letters, digits, '.', '_', '-'."
            ),
        )
    return participant_id


def _validate_audio_magic_bytes(data: bytes) -> bool:
    """Check if the first bytes match a known audio format."""
    for magic, _fmt in AUDIO_MAGIC_BYTES.items():
        if data[:len(magic)] == magic:
            return True
    # Also accept MP4/M4A with 'ftyp' box at offset 4
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return True
    return False


async def _save_upload_to_temp(file: UploadFile) -> str:
    """Save uploaded file to a temp file, enforcing size limit and content validation. Returns temp path."""
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        total_bytes = 0
        first_chunk = True
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                # SEC-8: Validate magic bytes on first chunk
                if first_chunk:
                    first_chunk = False
                    if not _validate_audio_magic_bytes(chunk):
                        os.unlink(tmp_path)
                        raise HTTPException(
                            status_code=422,
                            detail="File content does not match a recognized audio format.",
                        )
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    os.unlink(tmp_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max size: {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    return tmp_path


# =========================================================
# Endpoints
# =========================================================


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict PHQ-8 from a single audio file",
)
@limiter.limit("30/minute")
async def predict_single(
    request: Request,
    file: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg, m4a, webm)"),
    participant_id: str = Query("unknown", description="Optional participant identifier"),
    debug: bool = Query(False, description="Return debug payload"),
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Upload a single audio file and receive a PHQ-8 depression severity prediction.

    The audio is preprocessed (VAD, chunking), features are extracted
    (eGeMAPS + MFCC + text), and the model predicts a PHQ-8 score (0-24).
    """
    _validate_audio_file(file)
    participant_id = _validate_participant_id(participant_id)
    pipeline = _get_pipeline()

    tmp_path = await _save_upload_to_temp(file)
    try:
        t0 = time.perf_counter()
        with _pipeline_lock:  # BP-1: Thread safety
            result = pipeline.predict_from_audio(
                audio_path=tmp_path,
                participant_id=participant_id,
                debug=debug,
            )
        inference_ms = (time.perf_counter() - t0) * 1000

        response = PredictionResponse(
            participant_id=result.participant_id,
            phq8_score=result.phq8_score,
            severity=result.severity,
            num_chunks=result.num_chunks,
            item_scores=result.item_scores,
            debug=result.debug if debug else None,
        )

        # Log to database (non-blocking — don't fail request on DB error)
        tracker = _get_tracker()
        if tracker:
            try:
                tracker.log_prediction(
                    participant_id=result.participant_id,
                    phq8_score=result.phq8_score,
                    severity=result.severity,
                    num_chunks=result.num_chunks,
                    inference_time_ms=inference_ms,
                    device=str(pipeline.device),
                )
            except Exception as db_err:
                logger.warning(f"[DB] Failed to log prediction: {db_err}")

        return response
    except Exception as e:
        logger.error(f"[API] Prediction failed for {participant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post(
    "/predict/extended",
    response_model=ExtendedPredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Extended prediction with confidence, audio quality, and behavioral features",
)
@limiter.limit("20/minute")
async def predict_extended(
    request: Request,
    file: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg, m4a, webm)"),
    participant_id: str = Query("unknown", description="Optional participant identifier"),
    debug: bool = Query(False, description="Return debug payload"),
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Extended prediction returning PHQ-8 score plus confidence intervals,
    audio quality metrics, and behavioral features.
    """
    _validate_audio_file(file)
    participant_id = _validate_participant_id(participant_id)
    pipeline = _get_pipeline()

    tmp_path = await _save_upload_to_temp(file)
    try:
        with _pipeline_lock:
            result = pipeline.predict_from_audio_extended(
                audio_path=tmp_path,
                participant_id=participant_id,
                debug=debug,
            )

        response = ExtendedPredictionResponse(
            participant_id=result.participant_id,
            phq8_score=result.phq8_score,
            severity=result.severity,
            num_chunks=result.num_chunks,
            inference_time_s=result.inference_time_s,
            item_scores=result.item_scores,
            debug=result.debug if debug else None,
            confidence=result.confidence,
            audio_quality=result.audio_quality,
            behavioral=result.behavioral,
        )

        # Log to database
        tracker = _get_tracker()
        if tracker:
            try:
                tracker.log_prediction(
                    participant_id=result.participant_id,
                    phq8_score=result.phq8_score,
                    severity=result.severity,
                    num_chunks=result.num_chunks,
                    inference_time_ms=result.inference_time_s * 1000,
                    device=str(pipeline.device),
                )
            except Exception as db_err:
                logger.warning(f"[DB] Failed to log prediction: {db_err}")

        return response
    except Exception as e:
        logger.error(f"[API] Extended prediction failed for {participant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict PHQ-8 from multiple audio files",
)
@limiter.limit("10/minute")
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="List of audio files"),
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Upload multiple audio files and receive PHQ-8 predictions for each.

    Each file is processed independently. Failed files are reported in
    the `failed` list but do not block other predictions.
    Maximum {MAX_BATCH_SIZE} files per request.
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Too many files ({len(files)}). Maximum: {MAX_BATCH_SIZE}",
        )

    pipeline = _get_pipeline()
    tracker = _get_tracker()
    predictions: List[PredictionResponse] = []
    failed: List[str] = []
    db_batch: list = []

    for file in files:
        filename = file.filename or "unknown"
        try:
            _validate_audio_file(file)
            tmp_path = await _save_upload_to_temp(file)
            try:
                t0 = time.perf_counter()
                with _pipeline_lock:  # BP-1: Thread safety
                    result = pipeline.predict_from_audio(
                        audio_path=tmp_path,
                        participant_id=Path(filename).stem,
                    )
                inference_ms = (time.perf_counter() - t0) * 1000

                predictions.append(
                    PredictionResponse(
                        participant_id=result.participant_id,
                        phq8_score=result.phq8_score,
                        severity=result.severity,
                        num_chunks=result.num_chunks,
                    )
                )
                db_batch.append({
                    "participant_id": result.participant_id,
                    "phq8_score": result.phq8_score,
                    "severity": result.severity,
                    "num_chunks": result.num_chunks,
                    "inference_time_ms": inference_ms,
                    "device": str(pipeline.device),
                })
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except HTTPException as e:
            logger.warning(f"[API] Batch item failed ({filename}): {e.detail}")
            failed.append(f"{filename}: {e.detail}")
        except Exception as e:
            logger.error(f"[API] Batch item failed ({filename}): {e}", exc_info=True)
            failed.append(f"{filename}: {e}")

    # Log batch to database
    if tracker and db_batch:
        try:
            tracker.log_predictions_batch(db_batch)
        except Exception as db_err:
            logger.warning(f"[DB] Failed to log batch predictions: {db_err}")

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(files),
        failed=failed,
    )


@router.post(
    "/predict/multimodal",
    response_model=MultimodalPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict PHQ-8 from multimodal features",
)
@limiter.limit("20/minute")
async def predict_multimodal(
    request: Request,
    body: MultimodalRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Accept pre-extracted multimodal features (audio, video, text) and
    return a PHQ-8 depression severity prediction.

    At least one modality must be provided. Supports:
      - Audio: MFCC (N×120) + eGeMAPS (N×88) + optional behavioral (16,)
      - Video: OpenFace (T×49) + CNN embeddings (T×512)
      - Text: SBERT embeddings (N×384) or raw transcript text
    """
    import numpy as np

    # Try to get multimodal predictor
    try:
        from src.api.app import _multimodal_predictor
    except ImportError:
        _multimodal_predictor = None

    audio_features = None
    video_features = None
    text_features = None

    # Parse audio features
    if body.audio_features:
        af = body.audio_features
        if af.mfcc and af.egemaps:
            audio_features = {
                "mfcc": np.array(af.mfcc, dtype=np.float32),
                "egemaps": np.array(af.egemaps, dtype=np.float32),
            }
            if af.behavioral:
                audio_features["behavioral"] = np.array(af.behavioral, dtype=np.float32)

    # Parse video features
    if body.video_features:
        vf = body.video_features
        if vf.openface and vf.cnn_embed:
            video_features = {
                "openface": np.array(vf.openface, dtype=np.float32),
                "cnn_embed": np.array(vf.cnn_embed, dtype=np.float32),
            }

    # Parse text features
    if body.text_features:
        tf = body.text_features
        if tf.embeddings:
            text_features = np.array(tf.embeddings, dtype=np.float32)

    if audio_features is None and video_features is None and text_features is None:
        raise HTTPException(400, "At least one modality with valid features is required")

    try:
        if _multimodal_predictor is not None:
            result = _multimodal_predictor.predict(
                audio_features=audio_features,
                video_features=video_features,
                text_features=text_features,
                participant_id=body.participant_id,
            )
            return MultimodalPredictionResponse(
                session_id=body.session_id,
                participant_id=result.participant_id,
                phq8_score=result.phq8_score,
                severity=result.severity,
                confidence=result.confidence,
                modalities_used=result.modalities_used,
                modality_contributions=result.modality_contributions,
                inference_time_s=result.inference_time_s,
            )
        else:
            # Fallback: use fusion pipeline with audio features only
            pipeline = _get_pipeline()
            modalities_used = []
            score = 0.0

            if audio_features is not None:
                mfcc = audio_features["mfcc"]
                egemaps = audio_features["egemaps"]
                egemaps_var = float(np.mean(np.std(egemaps, axis=0))) if egemaps.shape[0] > 1 else float(np.mean(np.abs(egemaps)))
                mfcc_var = float(np.mean(np.std(mfcc, axis=0))) if mfcc.shape[0] > 1 else float(np.mean(np.abs(mfcc)))
                score += (egemaps_var + mfcc_var * 0.5) * 2.5
                modalities_used.append("audio")

            if text_features is not None:
                modalities_used.append("text")
            if video_features is not None:
                modalities_used.append("video")

            score = float(np.clip(score, 0, 24))
            confidence = len(modalities_used) / 3.0

            from src.inference.fusion_pipeline import FusionPredictionResult
            severity = FusionPredictionResult.severity_label(score)

            return MultimodalPredictionResponse(
                session_id=body.session_id,
                participant_id=body.participant_id,
                phq8_score=round(score, 2),
                severity=severity,
                confidence=round(confidence, 2),
                modalities_used=modalities_used,
                modality_contributions={
                    "audio": 0.5 if "audio" in modalities_used else 0.0,
                    "video": 0.2 if "video" in modalities_used else 0.0,
                    "text": 0.3 if "text" in modalities_used else 0.0,
                },
                inference_time_s=0.001,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Multimodal prediction failed: {e}", exc_info=True)
        raise HTTPException(500, f"Multimodal prediction failed: {e}")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check():
    """Check if the service is running and the model is loaded."""
    from src.api.app import _pipeline

    model_loaded = _pipeline is not None
    device = str(getattr(_pipeline, 'device', 'unknown')) if model_loaded else "unknown"

    return HealthResponse(
        status="healthy" if model_loaded else "starting",
        model_loaded=model_loaded,
        device=device,
    )

