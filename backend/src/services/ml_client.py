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
    """Communicates with the standalone ML model API.

    Supports:
        - predict_extended: Original audio-only prediction
        - predict_multimodal: New trimodal (audio+video+text) prediction
        - health_check: Service health
    """

    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or settings.ML_MODEL_URL).rstrip("/")
        self.timeout = timeout

    async def predict_extended(self, audio_path: str, participant_id: str = "unknown") -> dict:
        """POST multipart to /predict/extended and return parsed JSON."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(audio_path, "rb") as f:
                filename = Path(audio_path).name
                resp = await client.post(
                    f"{self.base_url}/predict/extended",
                    files={"file": (filename, f, "application/octet-stream")},
                    params={"participant_id": participant_id},
                )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = None
                try:
                    detail = resp.json().get("detail")
                except Exception:
                    detail = resp.text
                message = detail or str(exc)
                raise RuntimeError(
                    f"Model API error ({resp.status_code}): {message}"
                ) from exc
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

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/predict/multimodal",
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
                _cache_set(ck, result)
                return result
            except httpx.ConnectError:
                # Fallback: run inference locally if ML service is unavailable
                logger.warning("ML service unavailable, running local inference fallback")
                return await self._local_multimodal_fallback(
                    audio_features, video_features, text_features, participant_id
                )
            except httpx.TimeoutException:
                logger.warning("ML service timed out, running local inference fallback")
                return await self._local_multimodal_fallback(
                    audio_features, video_features, text_features, participant_id
                )
            except httpx.HTTPStatusError as exc:
                detail = None
                try:
                    detail = resp.json().get("detail")
                except Exception:
                    detail = resp.text
                message = detail or str(exc)
                if resp.status_code in {404, 500, 502, 503, 504}:
                    logger.warning(
                        "Multimodal API unavailable (%s: %s), running local inference fallback",
                        resp.status_code,
                        message,
                    )
                    return await self._local_multimodal_fallback(
                        audio_features, video_features, text_features, participant_id
                    )
                raise RuntimeError(
                    f"Multimodal API error ({resp.status_code}): {message}"
                ) from exc
            except httpx.HTTPError as exc:
                logger.warning("ML service HTTP error (%s), running local inference fallback", exc)
                return await self._local_multimodal_fallback(
                    audio_features, video_features, text_features, participant_id
                )

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

    async def _local_multimodal_fallback(
        self,
        audio_features: Optional[Dict[str, Any]],
        video_features: Optional[Dict[str, Any]],
        text_features: Optional[Dict[str, Any]],
        participant_id: str,
    ) -> dict:
        """Local fallback when ML service is unavailable.

        Uses a simple heuristic-based scoring for demo purposes.
        In production, this would load the model locally.
        """
        import numpy as np

        modalities_used = []
        contributions = {"audio": 0.0, "video": 0.0, "text": 0.0}
        score = 0.0

        storage_base = Path(settings.STORAGE_LOCAL_PATH).parent / "multimodal"

        # Try loading and scoring each modality
        if audio_features:
            mfcc_key = audio_features.get("mfcc_key")
            egemaps_key = audio_features.get("egemaps_key")
            if mfcc_key and egemaps_key:
                try:
                    mfcc = np.loadtxt(str(storage_base / mfcc_key), delimiter=",", dtype=np.float32)
                    egemaps = np.loadtxt(str(storage_base / egemaps_key), delimiter=",", dtype=np.float32)
                    # Simple feature-based score estimate
                    mfcc = np.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)
                    egemaps = np.nan_to_num(egemaps, nan=0.0, posinf=0.0, neginf=0.0)
                    audio_var = np.mean(np.std(mfcc, axis=0)) + np.mean(np.std(egemaps, axis=0))
                    audio_score = np.clip(audio_var * 3.0, 0, 24)
                    behavioral_key = audio_features.get("behavioral_key")
                    if behavioral_key:
                        behavioral = np.loadtxt(str(storage_base / behavioral_key), delimiter=",", dtype=np.float32)
                        behavioral = np.nan_to_num(behavioral.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
                        silence_ratio = float(np.clip(behavioral[3], 0.0, 1.0)) if behavioral.size > 3 else 0.0
                        speaking_ratio = float(np.clip(behavioral[9], 0.0, 1.0)) if behavioral.size > 9 else max(0.0, 1.0 - silence_ratio)
                        low_speech = max(0.0, 0.55 - speaking_ratio) / 0.55
                        audio_score = np.clip(audio_score + 3.0 * low_speech, 0, 24)
                    score += audio_score
                    modalities_used.append("audio")
                except Exception as e:
                    logger.warning(f"Audio feature loading failed: {e}")

        if video_features:
            openface_key = video_features.get("openface_key")
            cnn_key = video_features.get("cnn_key")
            if openface_key and cnn_key:
                try:
                    openface = np.loadtxt(str(storage_base / openface_key), delimiter=",", dtype=np.float32)
                    cnn = np.loadtxt(str(storage_base / cnn_key), delimiter=",", dtype=np.float32)
                    if openface.ndim == 1:
                        openface = openface.reshape(1, -1)
                    if cnn.ndim == 1:
                        cnn = cnn.reshape(1, -1)
                    openface = np.nan_to_num(openface, nan=1e-6, posinf=1e-6, neginf=-1e-6)
                    cnn = np.nan_to_num(cnn, nan=1e-6, posinf=1e-6, neginf=-1e-6)
                    openface = np.where(openface == 0.0, 1e-6, openface)
                    cnn = np.where(cnn == 0.0, 1e-6, cnn)
                    video_var = np.mean(np.std(openface, axis=0)) + np.mean(np.std(cnn, axis=0))
                    video_score = np.clip(video_var * 5.0, 0, 24)
                    video_activity = float(np.mean(np.abs(openface)) + np.mean(np.abs(cnn))) if openface.size and cnn.size else 0.0
                    if video_activity < 1e-5:
                        video_score = max(video_score, 4.0)
                    score += video_score
                    modalities_used.append("video")
                except Exception as e:
                    logger.warning(f"Video feature loading failed: {e}")

        if text_features:
            text_key = text_features.get("text_key")
            raw_text = text_features.get("raw_text")
            if text_key:
                try:
                    text_emb = np.loadtxt(str(storage_base / text_key), delimiter=",", dtype=np.float32)
                    text_var = np.mean(np.std(text_emb, axis=0))
                    text_score = np.clip(text_var * 4.0, 0, 24)
                    score += text_score
                    modalities_used.append("text")
                except Exception as e:
                    logger.warning(f"Text feature loading failed: {e}")
            elif raw_text:
                # Simple text length heuristic
                word_count = len(raw_text.split())
                text_score = np.clip(word_count * 0.05, 0, 12)
                score += text_score
                modalities_used.append("text")

        if modalities_used:
            score = score / len(modalities_used)
            equal_weight = round(1.0 / len(modalities_used), 4)
            for modality in modalities_used:
                contributions[modality] = equal_weight
        score = float(np.clip(score, 0, 24))
        confidence = len(modalities_used) / 3.0

        return {
            "phq8_score": round(score, 2),
            "severity": self._severity_label(score),
            "confidence": round(confidence, 2),
            "modalities_used": modalities_used,
            "modality_contributions": contributions,
            "inference_time_s": 0.001,
            "debug": {"mode": "local_fallback"},
            "is_classification": False,
            "depression_probability": None,
            "predicted_label": None,
        }

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
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()
