"""Async HTTP client for the ML model service — supports audio-only and multimodal."""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import httpx

from config.settings import get_settings

logger = logging.getLogger("mindscope")
settings = get_settings()


# ── In-process inference result cache ─────────────────────────────────────────
# Avoids re-sending identical feature sets to the ML service within a session.
# For multi-process deployments set REDIS_URL and the cache promotes to Redis.

_inference_cache: dict = {}


def _cache_key(payload: dict) -> str:
    """Stable SHA-256 hash of the JSON-serialised payload."""
    try:
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
    except Exception:
        return ""


def _cache_get(key: str) -> Optional[dict]:
    if not key:
        return None
    entry = _inference_cache.get(key)
    if entry is None:
        return None
    ttl = settings.ML_CACHE_TTL_SECONDS
    if (time.monotonic() - entry["ts"]) > ttl:
        _inference_cache.pop(key, None)
        return None
    logger.debug("ML cache HIT for key=%s...", key[:12])
    return entry["result"]


def _cache_set(key: str, result: dict) -> None:
    if not key:
        return
    _inference_cache[key] = {"result": result, "ts": time.monotonic()}
    # Evict oldest entries when the cache exceeds the configured limit
    max_entries = settings.ML_CACHE_MAX_ENTRIES
    if len(_inference_cache) > max_entries:
        oldest = sorted(_inference_cache.items(), key=lambda x: x[1]["ts"])
        for k, _ in oldest[: len(_inference_cache) - max_entries]:
            _inference_cache.pop(k, None)


def clear_inference_cache() -> int:
    """Flush the in-process cache.  Returns the number of entries removed."""
    count = len(_inference_cache)
    _inference_cache.clear()
    return count


class MLClient:
    """Communicates with the ML model API (Hugging Face Space).

    Supports:
        - predict_extended: Per-question audio scoring → POST /predict/audio
        - predict_multimodal: Trimodal inference → POST /predict/multimodal
        - health_check: Service liveness → GET /health
    """

    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base_url = (base_url or settings.ML_MODEL_URL).rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _extract_mfcc_features(audio_bytes: bytes, filename: str = "audio.webm") -> list:
        """Extract 39-dim MFCC+delta+delta2 features from raw audio bytes.

        The HF Space /predict/audio preprocessor expects this exact format:
        shape (T, 39) where T is the number of analysis frames.
        Server-side normalizer+PCA reduces 39→33 dims before inference.
        """
        import os
        import tempfile
        import numpy as np
        import librosa

        suffix = Path(filename).suffix.lower() or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        finally:
            os.unlink(tmp_path)

        if len(y) < sr * 0.5:
            raise RuntimeError(
                "Audio too short for feature extraction (< 0.5 s). "
                "Please re-record and speak for at least a few seconds."
            )

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta, delta2], axis=0).T  # (T, 39)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return features.tolist()

    async def predict_extended(
        self,
        audio_content: bytes,
        participant_id: str = "unknown",
        filename: str = "audio.webm",
    ) -> dict:
        """Extract MFCC features from raw audio bytes, POST JSON to /predict/audio.

        Feature extraction (39-dim MFCC+delta+delta2) happens on the backend.
        The HF Space /predict/audio endpoint applies normalizer+PCA (39→33)
        and runs inference.  This endpoint has always existed on the HF Space,
        unlike the /predict/audio/raw multipart endpoint.
        """
        url = f"{self.base_url}/predict/audio"
        logger.info("[ML] extracting MFCC features from %s (%d bytes)", filename, len(audio_content))
        t0 = time.monotonic()

        try:
            features = self._extract_mfcc_features(audio_content, filename)
        except Exception as exc:
            logger.error("[ML] MFCC extraction failed for %s: %s", filename, exc)
            raise RuntimeError(f"Audio feature extraction failed: {exc}") from exc

        payload = {"audio_features": features, "participant_id": participant_id}
        logger.info(
            "[ML] → POST %s (frames=%d, dims=%d)",
            url, len(features), len(features[0]) if features else 0,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload)
            except httpx.ConnectError as exc:
                logger.error("[ML] ConnectError → %s : %s", url, exc)
                raise RuntimeError(
                    f"Cannot reach ML service at {self.base_url}. "
                    "Check ML_MODEL_URL env var and that the HF Space is running."
                ) from exc
            except httpx.TimeoutException as exc:
                logger.error("[ML] Timeout after %.0fs → %s", self.timeout, url)
                raise RuntimeError(
                    f"ML service timed out after {self.timeout}s. "
                    "The HF Space may be cold-starting; please retry in a moment."
                ) from exc

        elapsed = time.monotonic() - t0
        logger.info("[ML] ← status=%d in %.2fs", resp.status_code, elapsed)
        if not resp.is_success:
            logger.error("[ML] ← FAILED status=%d body=%s", resp.status_code, resp.text[:500])
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = None
            try:
                detail = resp.json().get("detail")
            except Exception:
                detail = resp.text[:300]
            raise RuntimeError(f"ML model API error ({resp.status_code}): {detail or exc}") from exc
        return resp.json()

    async def predict_multimodal(
        self,
        session_id: str,
        audio_features: Optional[Dict[str, Any]] = None,
        video_features: Optional[Dict[str, Any]] = None,
        text_features: Optional[Dict[str, Any]] = None,
        participant_id: str = "unknown",
    ) -> dict:
        """POST multimodal features to /predict/multimodal.

        Sends feature file paths (storage keys) to the ML service.
        The ML service loads the features and runs trimodal inference.

        Args:
            session_id: Unique session identifier
            audio_features: Dict with mfcc_key, egemaps_key, behavioral_key
            video_features: Dict with openface_key, cnn_key
            text_features: Dict with text_key, raw_text
            participant_id: Patient/user identifier

        Returns:
            Dict with phq8_score, severity, confidence, modality_contributions, etc.
        """
        payload = {
            "session_id": session_id,
            "participant_id": participant_id,
        }

        storage_base = Path(settings.STORAGE_LOCAL_PATH).parent / "multimodal"
        if audio_features:
            audio_payload = self._load_audio_payload(storage_base, audio_features)
            if audio_payload:
                payload["audio_features"] = audio_payload
        if video_features:
            video_payload = self._load_video_payload(storage_base, video_features)
            if video_payload:
                payload["video_features"] = video_payload
        if text_features:
            text_payload = self._load_text_payload(storage_base, text_features)
            if text_payload:
                payload["text_features"] = text_payload

        if not any(k in payload for k in ("audio_features", "video_features", "text_features")):
            raise RuntimeError("No valid multimodal feature files were available for inference")

        # Cache check — skip network call if we have an identical recent result
        ck = _cache_key(payload)
        cached = _cache_get(ck)
        if cached is not None:
            return cached

        url = f"{self.base_url}/predict/multimodal"
        modalities_present = [m for m in ("audio_features", "video_features", "text_features") if m in payload]
        logger.info(
            "[ML] → POST %s (session=%s, modalities=%s)",
            url, payload.get("session_id", "?"), modalities_present,
        )
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload)
                elapsed = time.monotonic() - t0
                logger.info("[ML] ← status=%d in %.2fs", resp.status_code, elapsed)
                if not resp.is_success:
                    logger.error("[ML] ← FAILED status=%d body=%s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
                result = resp.json()
                _cache_set(ck, result)
                return result
            except httpx.ConnectError as exc:
                logger.error(
                    "[ML] ConnectError → %s : %s — check ML_MODEL_URL=%s",
                    url, exc, self.base_url,
                )
                raise RuntimeError(
                    f"Cannot reach ML service at {self.base_url}. "
                    "Check ML_MODEL_URL env var and that the HF Space is running."
                ) from exc
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - t0
                logger.error("[ML] Timeout after %.0fs → %s", elapsed, url)
                raise RuntimeError(
                    f"ML service timed out after {self.timeout}s. "
                    "The HF Space may be cold-starting; please retry in a moment."
                ) from exc
            except httpx.HTTPStatusError as exc:
                elapsed = time.monotonic() - t0
                detail = None
                try:
                    detail = resp.json().get("detail")
                except Exception:
                    detail = resp.text[:300]
                message = detail or str(exc)
                logger.error(
                    "[ML] ← error %d in %.2fs: %s",
                    resp.status_code, elapsed, message,
                )
                raise RuntimeError(
                    f"Multimodal ML API error ({resp.status_code}): {message}"
                ) from exc
            except httpx.HTTPError as exc:
                logger.error("[ML] HTTPError → %s : %s", url, exc)
                raise RuntimeError(f"ML service HTTP error: {exc}") from exc

    @staticmethod
    def _sanitize_array(path: Path, *, replace_zeros: bool = False) -> list:
        """Load a CSV feature file and return finite JSON-safe floats."""
        import numpy as np

        arr = np.loadtxt(str(path), delimiter=",", dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = np.nan_to_num(arr, nan=1e-6, posinf=1e-6, neginf=-1e-6).astype(np.float32)
        if replace_zeros:
            arr = np.where(arr == 0.0, 1e-6, arr).astype(np.float32)
            if arr.size and float(np.max(np.abs(arr))) <= 1e-6:
                base = np.linspace(1e-6, 1e-6 * arr.shape[1], arr.shape[1], dtype=np.float32)
                scales = np.arange(1, arr.shape[0] + 1, dtype=np.float32).reshape(-1, 1)
                arr = base.reshape(1, -1) * scales
        return arr.tolist()

    @staticmethod
    def _safe_feature_path(storage_base: Path, key: Optional[str]) -> Optional[Path]:
        """Resolve a storage key under multimodal storage, rejecting path escapes."""
        if not key:
            return None
        base = storage_base.resolve()
        path = (base / key).resolve()
        try:
            if not path.is_relative_to(base):
                logger.warning("Rejected feature path outside storage: %s", key)
                return None
        except AttributeError:
            if base not in path.parents and path != base:
                logger.warning("Rejected feature path outside storage: %s", key)
                return None
        if not path.exists():
            logger.warning("Feature file missing: %s", path)
            return None
        return path

    def _load_audio_payload(self, storage_base: Path, audio_features: Dict[str, Any]) -> Optional[list]:
        """Load combined audio features (MFCC+eGeMAPS = 62-dim) as a 2-D list.

        The ML server applies normalizer+PCA to reduce to the model's audio_dim=33.
        """
        import numpy as np
        # Prefer pre-concatenated combined file (saved by video_processor)
        combined_path = self._safe_feature_path(storage_base, audio_features.get("audio_combined_key"))
        if combined_path:
            return self._sanitize_array(combined_path)
        # Fallback: concatenate mfcc + egemaps on the fly
        mfcc_path = self._safe_feature_path(storage_base, audio_features.get("mfcc_key"))
        egemaps_path = self._safe_feature_path(storage_base, audio_features.get("egemaps_key"))
        if mfcc_path and egemaps_path:
            mfcc = np.array(self._sanitize_array(mfcc_path), dtype=np.float32)
            egemaps = np.array(self._sanitize_array(egemaps_path), dtype=np.float32)
            min_len = min(len(mfcc), len(egemaps))
            return np.concatenate([mfcc[:min_len], egemaps[:min_len]], axis=1).tolist()
        if mfcc_path:
            return self._sanitize_array(mfcc_path)
        return None

    def _load_video_payload(self, storage_base: Path, video_features: Dict[str, Any]) -> Optional[list]:
        """Load OpenFace features (49-dim) as a 2-D list.

        Training used 49-dim OpenFace features only. The ML server applies
        normalizer+PCA to reduce to model's video_dim=40.
        """
        openface_path = self._safe_feature_path(storage_base, video_features.get("openface_key"))
        if not openface_path:
            return None
        return self._sanitize_array(openface_path, replace_zeros=True)

    def _load_text_payload(self, storage_base: Path, text_features: Dict[str, Any]) -> Optional[list]:
        """Load text embeddings (384-dim) as a 2-D list.

        The ML server applies normalizer+PCA to reduce to model's text_dim=163.
        """
        text_path = self._safe_feature_path(storage_base, text_features.get("text_key"))
        if text_path:
            return self._sanitize_array(text_path)
        return None

    @staticmethod
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

    async def health_check(self) -> dict:
        """GET /health from the ML model service."""
        url = f"{self.base_url}/health"
        logger.info("[ML] → GET %s", url)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            logger.info("[ML] ← status=%d", resp.status_code)
            resp.raise_for_status()
            return resp.json()
