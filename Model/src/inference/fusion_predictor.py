"""
Fusion Model Predictor — DepressoSpeech

Loads the trained MultimodalFusion checkpoint and runs deterministic inference.
Handles text(384) + mfcc(120) + egemaps(88) + behavioral(16) features.
"""

import numpy as np
import hashlib
import torch
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple

from src.models.multimodal_fusion import MultimodalFusion

logger = logging.getLogger(__name__)


class FusionPredictor:
    """
    Wraps MultimodalFusion for deterministic inference.

    Prediction flow:
        1. Load pretrained text backbone + fusion checkpoint
        2. Receive per-segment text, mfcc, egemaps + interview-level behavioral features
        3. StatsPool → BN → text_pred + residual → PHQ-8 score
    """

    def __init__(
        self,
        fusion_checkpoint: Union[str, Path],
        text_checkpoint: Union[str, Path] = "checkpoints/best_model.pt",
        device: str = "auto",
    ):
        self.device = self._resolve_device(device)

        # Load fusion weights first so model topology can be inferred safely.
        ckpt = torch.load(
            str(fusion_checkpoint), map_location=self.device, weights_only=False
        )
        state = ckpt.get("model_state_dict", {})

        text_bn_weight = state.get("text_bn.weight")
        text_bn_dim = int(text_bn_weight.shape[0]) if text_bn_weight is not None else 768
        if text_bn_dim == 384:
            stats_mode = "mean"
        elif text_bn_dim == 768:
            stats_mode = "mean_std"
        elif text_bn_dim == 1536:
            stats_mode = "mean_std_min_max"
        else:
            stats_mode = "mean_std"

        residual_bn_weight = state.get("residual_bn.weight")
        residual_dim = int(residual_bn_weight.shape[0]) if residual_bn_weight is not None else 80
        if residual_dim in (432, 416):
            use_audio_encoder = False
            use_behavioral = residual_dim == 432
        elif residual_dim in (80, 64):
            use_audio_encoder = True
            use_behavioral = residual_dim == 80
        else:
            use_audio_encoder = True
            use_behavioral = True

        # Build model
        self.model = MultimodalFusion(
            stats_mode=stats_mode,
            use_behavioral=use_behavioral,
            use_audio_encoder=use_audio_encoder,
        ).to(self.device)
        try:
            self.model.load_pretrained_text(str(text_checkpoint))
        except (FileNotFoundError, KeyError, RuntimeError, OSError) as exc:
            logger.warning(f"Continuing without text checkpoint preload: {exc}")

        # Load fusion weights
        load_result = self.model.load_state_dict(state, strict=False)
        if load_result.missing_keys:
            logger.warning(f"Fusion checkpoint missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.warning(f"Fusion checkpoint unexpected keys: {load_result.unexpected_keys}")
        self.model.eval()

        val_ccc = ckpt.get("val_ccc")
        if val_ccc is None:
            val_ccc = ckpt.get("metrics", {}).get("ccc")
        state_keys = set(ckpt.get("model_state_dict", {}).keys())
        has_audio_branch = any(
            key.startswith(("audio_encoder.", "audio_pool.", "residual_"))
            for key in state_keys
        )
        self.metadata = {
            "epoch": ckpt.get("epoch", -1),
            "val_ccc": val_ccc,
            "model_version": self._compute_version(ckpt),
            "has_audio_branch": has_audio_branch,
        }
        self.use_fusion = bool(use_audio_encoder)

        total = sum(p.numel() for p in self.model.parameters())
        ccc_label = f"{self.metadata['val_ccc']:.4f}" if self.metadata["val_ccc"] is not None else "unknown"
        logger.info(
            f"FusionPredictor loaded: epoch={self.metadata['epoch']}, "
            f"val_ccc={ccc_label}, mode={'fusion' if self.use_fusion else 'text_only'}, "
            f"params={total:,}, device={self.device}"
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _compute_version(ckpt: dict) -> str:
        state = ckpt.get("model_state_dict", {})
        h = hashlib.sha256()
        for key in sorted(state.keys()):
            h.update(key.encode())
            h.update(state[key].cpu().numpy().tobytes())
        return h.hexdigest()[:12]

    @torch.no_grad()
    def predict(
        self,
        text_features: np.ndarray,
        mfcc_features: np.ndarray,
        egemaps_features: np.ndarray,
        behavioral: Optional[np.ndarray] = None,
        normalize_text: bool = True,
        text_scale: float = 0.55,
    ) -> float:
        """
        Predict PHQ-8 score from aligned per-segment features.

        Args:
            text_features:    (N, 384) per-segment text embeddings
            mfcc_features:    (N, 120) per-segment MFCC features
            egemaps_features: (N, 88)  per-segment eGeMAPS features
            behavioral:       (16,) interview-level behavioral features, or None
            normalize_text:   Whether to L2-normalize text. Set False if already normalized.
            text_scale:       Weight for text branch contribution [0,1]. Default 0.6
                              reduces text dominance so audio features contribute more.

        Returns:
            PHQ-8 score clamped to [0, 24]
        """
        return self._predict_components(
            text_features=text_features,
            mfcc_features=mfcc_features,
            egemaps_features=egemaps_features,
            behavioral=behavioral,
            normalize_text=normalize_text,
            text_scale=text_scale,
        )[0]

    @torch.no_grad()
    def predict_with_debug(
        self,
        text_features: np.ndarray,
        mfcc_features: np.ndarray,
        egemaps_features: np.ndarray,
        behavioral: Optional[np.ndarray] = None,
        normalize_text: bool = True,
        text_scale: float = 0.55,
    ) -> Tuple[float, Dict[str, Any]]:
        score, components = self._predict_components(
            text_features=text_features,
            mfcc_features=mfcc_features,
            egemaps_features=egemaps_features,
            behavioral=behavioral,
            normalize_text=normalize_text,
            text_scale=text_scale,
        )
        audio_signal = components["audio_signal"]
        debug = {
            "model_mode": components["model_mode"],
            "has_audio_branch": self.metadata.get("has_audio_branch", False),
            "val_ccc": self.metadata.get("val_ccc"),
            "audio_signal": round(float(audio_signal), 4),
            "audio_score": round(float(components["audio_score"]), 4),
            "text_score": round(float(components["text_score"]), 4),
            "multimodal_score": round(float(components["multimodal_score"]), 4),
            "final_score": round(float(score), 4),
            "text_scale": text_scale,
            "item_scores": self._derive_item_scores(score, audio_signal, behavioral),
            "audio_features": {
                "mfcc_mean": round(float(np.nan_to_num(mfcc_features, nan=0.0).mean()), 4),
                "mfcc_std": round(float(np.nan_to_num(mfcc_features, nan=0.0).std()), 4),
                "egemaps_mean": round(float(np.nan_to_num(egemaps_features, nan=0.0).mean()), 4),
                "egemaps_std": round(float(np.nan_to_num(egemaps_features, nan=0.0).std()), 4),
            },
            "behavioral_features": {
                "pause_signal": round(float(np.clip(behavioral[3], 0.0, 3.0) / 3.0), 4) if behavioral is not None and behavioral.size >= 4 else 0.0,
                "long_pause": round(float(np.clip(behavioral[6], 0.0, 1.0)), 4) if behavioral is not None and behavioral.size >= 7 else 0.0,
                "speaking_signal": round(float(np.clip(1.0 - behavioral[9], 0.0, 1.0)), 4) if behavioral is not None and behavioral.size >= 10 else 0.0,
            },
            "feature_shapes": {
                "text": list(text_features.shape),
                "mfcc": list(mfcc_features.shape),
                "egemaps": list(egemaps_features.shape),
                "behavioral": list(behavioral.shape) if behavioral is not None else [16],
            },
        }
        return score, debug

    def _predict_components(
        self,
        text_features: np.ndarray,
        mfcc_features: np.ndarray,
        egemaps_features: np.ndarray,
        behavioral: Optional[np.ndarray] = None,
        normalize_text: bool = True,
        text_scale: float = 0.55,
    ) -> Tuple[float, Dict[str, Any]]:
        if normalize_text:
            norms = np.linalg.norm(text_features, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            text_features = text_features / norms

        self._validate_features(text_features, mfcc_features, egemaps_features)

        text_t = torch.from_numpy(text_features.astype(np.float32)).unsqueeze(0).to(self.device)
        N = text_features.shape[0]
        mask = torch.ones(1, N, dtype=torch.bool, device=self.device)

        text_pred = self.model.predict_text_only(text_t, mask)
        text_score = float(text_pred.squeeze().clamp(0.0, 24.0).item())

        mfcc_norm = self._feature_normalize(mfcc_features)
        egemaps_norm = self._feature_normalize(egemaps_features)
        mfcc_t = torch.from_numpy(mfcc_norm.astype(np.float32)).unsqueeze(0).to(self.device)
        egemaps_t = torch.from_numpy(egemaps_norm.astype(np.float32)).unsqueeze(0).to(self.device)

        if behavioral is not None:
            behavioral_t = torch.from_numpy(behavioral.astype(np.float32)).unsqueeze(0).to(self.device)
        else:
            behavioral_t = torch.zeros(1, 16, device=self.device)

        if self.use_fusion:
            multimodal_pred = self.model(text_t, mfcc_t, egemaps_t, mask, behavioral_t, text_scale=text_scale)
            multimodal_score = float(multimodal_pred.squeeze().clamp(0.0, 24.0).item())
        else:
            multimodal_score = text_score

        audio_signal = self._audio_signal(mfcc_norm, egemaps_norm, behavioral)
        audio_score = float(np.clip(audio_signal * 24.0, 0.0, 24.0))

        if self.use_fusion:
            final_score = (0.45 * multimodal_score) + (0.55 * audio_score)
        else:
            final_score = (0.35 * text_score) + (0.65 * audio_score)

        final_score = float(np.clip(final_score, 0.0, 24.0))
        debug = {
            "model_mode": "fusion" if self.use_fusion else "text_plus_audio",
            "text_score": text_score,
            "multimodal_score": multimodal_score,
            "audio_score": audio_score,
            "audio_signal": audio_signal,
            "final_score": final_score,
        }
        return final_score, debug

    @staticmethod
    def _feature_normalize(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Per-session (within-recording) z-score normalization.

        Subtracts the session mean and divides by session std so the model
        receives relative feature patterns rather than absolute speaker-level
        values. This prevents a calm-voiced speaker from always scoring low
        and a loud speaker from always scoring high.

        Args:
            features: (N, D) array of raw per-chunk features
            eps: small constant to avoid division by zero

        Returns:
            (N, D) session-normalized features
        """
        if features.shape[0] < 2:
            return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        normalized = (features - mean) / std
        return np.nan_to_num(
            normalized.astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    @staticmethod
    def _validate_features(text: np.ndarray, mfcc: np.ndarray, egemaps: np.ndarray) -> None:
        if text.ndim != 2 or text.shape[1] != 384:
            raise ValueError(f"Expected text features (N,384), got {text.shape}")
        if mfcc.ndim != 2 or mfcc.shape[1] != 120:
            raise ValueError(f"Expected MFCC features (N,120), got {mfcc.shape}")
        if egemaps.ndim != 2 or egemaps.shape[1] != 88:
            raise ValueError(f"Expected eGeMAPS features (N,88), got {egemaps.shape}")
        if not (text.shape[0] == mfcc.shape[0] == egemaps.shape[0]):
            raise ValueError(
                f"Feature count mismatch: text={text.shape[0]}, mfcc={mfcc.shape[0]}, egemaps={egemaps.shape[0]}"
            )
        if not np.isfinite(mfcc).all() or not np.isfinite(egemaps).all() or not np.isfinite(text).all():
            raise ValueError("Non-finite values detected in model features")

    @staticmethod
    def _safe_mean(value: np.ndarray) -> float:
        if value.size == 0:
            return 0.0
        return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).mean())

    def _audio_signal(
        self,
        mfcc_features: np.ndarray,
        egemaps_features: np.ndarray,
        behavioral: Optional[np.ndarray],
    ) -> float:
        mfcc_var = float(np.mean(np.std(np.nan_to_num(mfcc_features, nan=0.0), axis=0))) if mfcc_features.size else 0.0
        egemaps_var = float(np.mean(np.std(np.nan_to_num(egemaps_features, nan=0.0), axis=0))) if egemaps_features.size else 0.0
        audio_flatness = 1.0 - np.tanh((mfcc_var + egemaps_var) / 8.0)

        if behavioral is None or behavioral.size < 16:
            pause_signal = 0.0
            speaking_signal = 0.0
            turn_brevity = 0.0
            long_pause = 0.0
        else:
            pause_signal = float(np.clip(behavioral[3], 0.0, 3.0) / 3.0)
            long_pause = float(np.clip(behavioral[6], 0.0, 1.0))
            speaking_signal = float(np.clip(1.0 - behavioral[9], 0.0, 1.0))
            turn_brevity = float(np.clip(behavioral[13], 0.0, 1.0))

        audio_signal = (
            0.34 * pause_signal
            + 0.24 * long_pause
            + 0.20 * speaking_signal
            + 0.12 * turn_brevity
            + 0.10 * audio_flatness
        )
        return float(np.clip(audio_signal, 0.0, 1.0))

    def _audio_bias(
        self,
        mfcc_features: np.ndarray,
        egemaps_features: np.ndarray,
        behavioral: Optional[np.ndarray],
    ) -> float:
        audio_signal = self._audio_signal(mfcc_features, egemaps_features, behavioral)
        return (audio_signal - 0.45) * 8.0

    @staticmethod
    def _derive_item_scores(
        total_score: float,
        audio_signal: float,
        behavioral: Optional[np.ndarray],
    ) -> list[int]:
        base = float(np.clip(total_score / 8.0, 0.0, 3.0))
        audio_boost = float(np.clip((audio_signal - 0.5) * 2.0, -1.0, 1.0))
        modifiers = np.array([0.22, 0.25, 0.12, 0.18, 0.10, 0.28, 0.20, 0.25], dtype=np.float32)
        if behavioral is not None and behavioral.size >= 16:
            modifiers += np.array([
                behavioral[3],
                behavioral[6],
                behavioral[9],
                behavioral[13],
                behavioral[15],
                behavioral[0],
                behavioral[1],
                behavioral[2],
            ], dtype=np.float32) * 0.05
        item_scores = np.clip(np.rint(base + (modifiers * audio_boost)), 0, 3).astype(int)
        return item_scores.tolist()
