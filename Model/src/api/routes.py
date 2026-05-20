from typing import Optional, Dict, Any
import hashlib
import io
import json
import time
import logging
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, BackgroundTasks, UploadFile
from src.api.schemas import AudioPayload, VideoPayload, TextPayload, MultimodalPayload, PredictionResponse, HealthResponse
from src.features.text_features import TextFeatureExtractor
from src.inference.inferencer import ModelV2Inferencer

router = APIRouter()
_inferencer: Optional[ModelV2Inferencer] = None
_text_extractor: Optional[TextFeatureExtractor] = None
_preprocessors: Dict[str, Any] = {}

logger = logging.getLogger(__name__)

# ── Route-level prediction cache (TTL = 1 hour, max 200 entries) ──────────────
_pred_cache: dict = {}
_PRED_CACHE_TTL = 3600
_PRED_CACHE_MAX = 200


def _pred_cache_key(payload_dict: dict) -> str:
    try:
        raw = json.dumps(payload_dict, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
    except Exception:
        return ""


def _pred_cache_get(key: str) -> Optional[dict]:
    entry = _pred_cache.get(key)
    if entry and (time.monotonic() - entry["ts"]) < _PRED_CACHE_TTL:
        return entry["v"]
    return None


def _pred_cache_set(key: str, value: dict) -> None:
    _pred_cache[key] = {"v": value, "ts": time.monotonic()}
    if len(_pred_cache) > _PRED_CACHE_MAX:
        oldest = sorted(_pred_cache.items(), key=lambda x: x[1]["ts"])
        for k, _ in oldest[: len(_pred_cache) - _PRED_CACHE_MAX]:
            _pred_cache.pop(k, None)


def register_inferencer(
    inferencer: ModelV2Inferencer,
    text_extractor: Optional[TextFeatureExtractor] = None,
    preprocessors: Optional[Dict[str, Any]] = None,
) -> None:
    global _inferencer, _text_extractor, _preprocessors
    _inferencer = inferencer
    _text_extractor = text_extractor
    if preprocessors:
        _preprocessors = preprocessors
        logger.info("Registered preprocessors for modalities: %s", list(preprocessors.keys()))


def _apply_preprocessor(features_list: Optional[list], modality: str) -> Optional[list]:
    """Apply normalizer + PCA to raw features using the training preprocessors."""
    if not features_list or modality not in _preprocessors:
        return features_list
    preproc = _preprocessors[modality]
    try:
        arr = np.array(features_list, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        normalizer = preproc.get("normalizer") if isinstance(preproc, dict) else getattr(preproc, "normalizer", None)
        pca = preproc.get("pca") if isinstance(preproc, dict) else getattr(preproc, "pca", None)
        if normalizer is not None:
            arr = normalizer.transform(arr)
        if pca is not None:
            arr = pca.transform(arr)
        return arr.tolist()
    except Exception as exc:
        logger.warning("Preprocessing failed for %s (%s); passing features as-is", modality, exc)
        return features_list


def get_inferencer() -> Optional[ModelV2Inferencer]:
    return _inferencer


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_inferencer is not None,
        description="ModelV2 inference service is available." if _inferencer else "Model is not loaded yet.",
    )


@router.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio(payload: AudioPayload, background_tasks: BackgroundTasks) -> PredictionResponse:
    if _inferencer is None:
        raise HTTPException(status_code=503, detail="Inference model is not initialized.")
    if not payload.audio_features:
        raise HTTPException(status_code=400, detail="Audio features are required.")
    start = time.perf_counter()
    sample = {
        "audio": payload.audio_features,
        "audio_mask": [True] * len(payload.audio_features),
    }
    result = _inferencer.predict_single(sample)
    duration = time.perf_counter() - start
    result.setdefault("metadata", {})
    result["processing_details"] = {"total_seconds": duration, "stages": {"inference": duration}}
    logger.info("predict_audio finished: duration=%.3fs classification=%.4f confidence=%.4f", duration, result.get("classification"), result.get("confidence"))
    return PredictionResponse(**result)


@router.post("/predict/video", response_model=PredictionResponse)
async def predict_video(payload: VideoPayload) -> PredictionResponse:
    if _inferencer is None:
        raise HTTPException(status_code=503, detail="Inference model is not initialized.")
    if not payload.video_features:
        raise HTTPException(status_code=400, detail="Video features are required.")
    start = time.perf_counter()
    sample = {
        "video": payload.video_features,
        "video_mask": [True] * len(payload.video_features),
    }
    result = _inferencer.predict_single(sample)
    duration = time.perf_counter() - start
    result.setdefault("metadata", {})
    result["processing_details"] = {"total_seconds": duration, "stages": {"inference": duration}}
    logger.info("predict_video finished: duration=%.3fs classification=%.4f confidence=%.4f", duration, result.get("classification"), result.get("confidence"))
    return PredictionResponse(**result)


@router.post("/predict/text", response_model=PredictionResponse)
async def predict_text(payload: TextPayload) -> PredictionResponse:
    if _inferencer is None:
        raise HTTPException(status_code=503, detail="Inference model is not initialized.")
    if payload.transcript is None and not payload.chunked_transcript:
        raise HTTPException(status_code=400, detail="Transcript or chunked_transcript is required.")
    if _text_extractor is None:
        raise HTTPException(status_code=503, detail="Text feature extractor is not configured.")
    chunks = payload.chunked_transcript or [payload.transcript]
    start_total = time.perf_counter()
    # measure text feature extraction
    t0 = time.perf_counter()
    features = _text_extractor.encode(chunks)
    t_text = time.perf_counter() - t0
    # Validate feature dimensionality matches model expectation
    expected_dim = _inferencer.model.text_input_dim if _inferencer else None
    if expected_dim is not None:
        if features.ndim == 1:
            feat_dim = features.shape[0]
        else:
            feat_dim = features.shape[-1]
        if feat_dim != expected_dim:
            raise HTTPException(status_code=400, detail=f"Text embedding dim ({feat_dim}) does not match model expected dim ({expected_dim}). Ensure the saved text preprocessor (PCA/normalizer) is available in checkpoints/scalers.")

    sample = {
        "text": features.tolist(),
        "text_mask": [True] * len(chunks),
    }
    # measure model inference
    t1 = time.perf_counter()
    result = _inferencer.predict_single(sample)
    t_infer = time.perf_counter() - t1

    total = time.perf_counter() - start_total
    result.setdefault("metadata", {})
    result["processing_details"] = {"total_seconds": total, "stages": {"text_extraction": t_text, "inference": t_infer}}
    logger.info("predict_text finished: total=%.3fs text=%.3fs inference=%.3fs classification=%.4f", total, t_text, t_infer, result.get("classification"))
    return PredictionResponse(**result)


@router.post("/predict/multimodal", response_model=PredictionResponse)
async def predict_multimodal(payload: MultimodalPayload) -> PredictionResponse:
    if _inferencer is None:
        raise HTTPException(status_code=503, detail="Inference model is not initialized.")
    # Check route-level cache first
    cache_key = _pred_cache_key(payload.model_dump())
    cached = _pred_cache_get(cache_key)
    if cached is not None:
        logger.debug("predict_multimodal cache HIT key=%s...", cache_key[:12])
        return PredictionResponse(**cached)
    start = time.perf_counter()
    audio = _apply_preprocessor(payload.audio_features, "audio")
    video = _apply_preprocessor(payload.video_features, "video")
    text  = _apply_preprocessor(payload.text_features,  "text")
    sample = {
        "audio": audio,
        "video": video,
        "text": text,
        "audio_mask": payload.audio_mask if payload.audio_mask is not None else ([True] * len(audio) if audio else [False]),
        "video_mask": payload.video_mask if payload.video_mask is not None else ([True] * len(video) if video else [False]),
        "text_mask":  payload.text_mask  if payload.text_mask  is not None else ([True] * len(text)  if text  else [False]),
    }
    result = _inferencer.predict_single(sample)
    duration = time.perf_counter() - start
    result.setdefault("metadata", {})
    result["processing_details"] = {"total_seconds": duration, "stages": {"inference": duration}}
    _pred_cache_set(cache_key, result)
    logger.info("predict_multimodal finished: duration=%.3fs classification=%.4f confidence=%.4f modality_scores=%s", duration, result.get("classification"), result.get("confidence"), result.get("modality_scores"))
    return PredictionResponse(**result)


@router.post("/predict/question", response_model=PredictionResponse)
async def predict_question(payload: MultimodalPayload) -> PredictionResponse:
    if _inferencer is None:
        raise HTTPException(status_code=503, detail="Inference model is not initialized.")
    start = time.perf_counter()
    audio = _apply_preprocessor(payload.audio_features, "audio")
    video = _apply_preprocessor(payload.video_features, "video")
    text  = _apply_preprocessor(payload.text_features,  "text")
    sample = {
        "audio": audio,
        "video": video,
        "text": text,
        "audio_mask": payload.audio_mask if payload.audio_mask is not None else ([True] * len(audio) if audio else [False]),
        "video_mask": payload.video_mask if payload.video_mask is not None else ([True] * len(video) if video else [False]),
        "text_mask":  payload.text_mask  if payload.text_mask  is not None else ([True] * len(text)  if text  else [False]),
    }
    result = _inferencer.predict_single(sample)
    duration = time.perf_counter() - start
    result.setdefault("metadata", {})
    result["processing_details"] = {"total_seconds": duration, "stages": {"inference": duration}}
    logger.info("predict_question finished: duration=%.3fs classification=%.4f", duration, result.get("classification"))
    return PredictionResponse(**result)


@router.post("/predict/audio/raw", response_model=PredictionResponse)
async def predict_audio_raw(
    file: UploadFile = File(...),
    participant_id: str = Form("unknown"),
) -> PredictionResponse:
    """Accept a raw audio file upload, extract MFCC features with librosa, and run inference.

    Used by the backend's predict_extended() call for per-question audio scoring.
    Feature pipeline: MFCC(13) + delta + delta2 → [T, 39] → normalizer+PCA → [T, audio_dim].
    """
    if _inferencer is None:
        raise HTTPException(status_code=503, detail="Inference model is not initialized.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    try:
        import librosa
        waveform, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {exc}")

    if len(waveform) < 320:
        raise HTTPException(status_code=400, detail="Audio too short — need at least 20ms of audio.")

    waveform = librosa.util.normalize(waveform)
    mfcc   = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=13)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2]).T.astype(np.float32)  # [T, 39]
    features = np.nan_to_num(features, nan=1e-6, posinf=1e-6, neginf=-1e-6)

    audio_features = _apply_preprocessor(features.tolist(), "audio")

    # Ensure output dim matches model's audio_input_dim regardless of preprocessor outcome
    target_dim = _inferencer.model.audio_input_dim
    arr = np.array(audio_features, dtype=np.float32)
    if arr.shape[-1] != target_dim:
        logger.warning(
            "predict_audio_raw: dim mismatch after preprocessing — got %d, expected %d; adjusting",
            arr.shape[-1], target_dim,
        )
        if arr.shape[-1] > target_dim:
            arr = arr[:, :target_dim]
        else:
            pad = np.zeros((arr.shape[0], target_dim - arr.shape[-1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=-1)
        audio_features = arr.tolist()

    start = time.perf_counter()
    sample = {
        "audio": audio_features,
        "audio_mask": [True] * len(audio_features),
    }
    result = _inferencer.predict_single(sample)
    duration = time.perf_counter() - start
    result.setdefault("metadata", {})
    result["processing_details"] = {
        "total_seconds": duration,
        "stages": {"inference": duration},
        "participant_id": participant_id,
        "audio_frames": len(audio_features),
    }
    logger.info(
        "predict_audio_raw finished: frames=%d duration=%.3fs classification=%.4f confidence=%.4f",
        len(audio_features), duration, result.get("classification"), result.get("confidence"),
    )
    return PredictionResponse(**result)
