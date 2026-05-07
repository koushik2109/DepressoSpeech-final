"""Async HTTP client for the ML model service — supports audio-only and multimodal."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import httpx

from config.settings import get_settings

logger = logging.getLogger("mindscope")
settings = get_settings()


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

        if audio_features:
            payload["audio_features"] = audio_features
        if video_features:
            payload["video_features"] = video_features
        if text_features:
            payload["text_features"] = text_features

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/predict/multimodal",
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.ConnectError:
                # Fallback: run inference locally if ML service is unavailable
                logger.warning("ML service unavailable, running local inference fallback")
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
                raise RuntimeError(
                    f"Multimodal API error ({resp.status_code}): {message}"
                ) from exc

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
                    audio_var = np.mean(np.std(mfcc, axis=0)) + np.mean(np.std(egemaps, axis=0))
                    audio_score = np.clip(audio_var * 3.0, 0, 24)
                    score += audio_score * 0.4
                    contributions["audio"] = 0.4
                    modalities_used.append("audio")
                except Exception as e:
                    logger.warning(f"Audio feature loading failed: {e}")

        if video_features:
            openface_key = video_features.get("openface_key")
            if openface_key:
                try:
                    openface = np.loadtxt(str(storage_base / openface_key), delimiter=",", dtype=np.float32)
                    video_var = np.mean(np.std(openface, axis=0))
                    video_score = np.clip(video_var * 5.0, 0, 24)
                    score += video_score * 0.3
                    contributions["video"] = 0.3
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
                    score += text_score * 0.3
                    contributions["text"] = 0.3
                    modalities_used.append("text")
                except Exception as e:
                    logger.warning(f"Text feature loading failed: {e}")
            elif raw_text:
                # Simple text length heuristic
                word_count = len(raw_text.split())
                text_score = np.clip(word_count * 0.05, 0, 12)
                score += text_score * 0.3
                contributions["text"] = 0.3
                modalities_used.append("text")

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
