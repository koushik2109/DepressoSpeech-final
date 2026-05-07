"""
Trimodal Inference Pipeline — DepressoSpeech V2

End-to-end: pre-extracted features → normalize → trimodal fusion → PHQ-8 score.

Supports:
    - Audio features (eGeMAPS, MFCC) from CSV
    - Video features (OpenFace, CNN embeddings) from CSV
    - Text features (SBERT embeddings) from CSV or raw text
    - Any combination of modalities (graceful degradation)
"""

import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field

import torch

from src.models.multimodal_fusion_v2 import TrimodalFusionModel, FusionOutput

logger = logging.getLogger(__name__)


@dataclass
class MultimodalPredictionResult:
    """Structured result from multimodal prediction."""
    phq8_score: float
    severity: str
    confidence: float = 0.0
    modalities_used: List[str] = field(default_factory=list)
    modality_contributions: Dict[str, float] = field(default_factory=dict)
    inference_time_s: float = 0.0
    participant_id: str = "unknown"
    debug: Optional[Dict[str, Any]] = None

    @staticmethod
    def severity_label(score: float) -> str:
        if score < 5:
            return "none/minimal"
        elif score < 10:
            return "mild"
        elif score < 15:
            return "moderate"
        elif score < 20:
            return "moderately severe"
        else:
            return "severe"


class MultimodalPredictor:
    """
    Wraps TrimodalFusionModel for production inference.

    Handles feature loading from CSV/arrays, normalization, and missing modality.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        fusion_dim: int = 128,
        openface_dim: int = 49,
        cnn_dim: int = 512,
    ):
        self.device = self._resolve_device(device)
        self.fusion_dim = fusion_dim

        # Build model
        self.model = TrimodalFusionModel(
            fusion_dim=fusion_dim,
            num_attention_heads=4,
            stats_mode="mean_std",
            dropout=0.1,
            modality_dropout=0.0,  # No dropout at inference
            openface_dim=openface_dim,
            cnn_dim=cnn_dim,
        ).to(self.device)

        # Load checkpoint if available
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            load_result = self.model.load_state_dict(state, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys: {load_result.missing_keys}")
            logger.info(
                f"Loaded trimodal checkpoint from {checkpoint_path}, "
                f"epoch={ckpt.get('epoch', '?')}"
            )
        else:
            logger.info("Trimodal model initialized without checkpoint (random weights)")

        self.model.eval()
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"MultimodalPredictor ready: {total:,} params, device={self.device}")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _normalize_features(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Per-session z-score normalization."""
        features = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if features.shape[0] < 2:
            return features
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        return ((features - mean) / std).astype(np.float32)

    @staticmethod
    def _load_csv_features(path: str) -> Optional[np.ndarray]:
        """Load features from CSV file. Returns None if file doesn't exist."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = np.loadtxt(str(p), delimiter=",", dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        """Convert numpy array to batched tensor on device."""
        return torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.device)

    def _make_mask(self, length: int) -> torch.Tensor:
        return torch.ones(1, length, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def predict(
        self,
        # Audio features (pre-extracted)
        audio_features: Optional[Dict[str, np.ndarray]] = None,
        # Video features (pre-extracted)
        video_features: Optional[Dict[str, np.ndarray]] = None,
        # Text features (pre-extracted embeddings)
        text_features: Optional[np.ndarray] = None,
        # Metadata
        participant_id: str = "unknown",
        debug: bool = False,
    ) -> MultimodalPredictionResult:
        """
        Predict PHQ-8 from pre-extracted multimodal features.

        Args:
            audio_features: Dict with keys "mfcc" (N,120), "egemaps" (N,88),
                           optionally "behavioral" (16,)
            video_features: Dict with keys "openface" (T,49), "cnn_embed" (T,512)
            text_features: (N,384) SBERT embeddings
            participant_id: Identifier for logging
            debug: Include debug info in result

        Returns:
            MultimodalPredictionResult
        """
        t_start = time.perf_counter()
        modalities_used = []
        kwargs = {}

        # ─── Process Audio ───
        if audio_features is not None:
            mfcc = audio_features.get("mfcc")
            egemaps = audio_features.get("egemaps")
            behavioral = audio_features.get("behavioral")

            if mfcc is not None and egemaps is not None:
                mfcc = self._normalize_features(mfcc)
                egemaps = self._normalize_features(egemaps)

                # Align chunk counts
                min_n = min(mfcc.shape[0], egemaps.shape[0])
                mfcc = mfcc[:min_n]
                egemaps = egemaps[:min_n]

                kwargs["mfcc"] = self._to_tensor(mfcc)
                kwargs["egemaps"] = self._to_tensor(egemaps)
                kwargs["audio_mask"] = self._make_mask(min_n)

                if behavioral is not None:
                    kwargs["behavioral"] = torch.from_numpy(
                        behavioral.astype(np.float32)
                    ).unsqueeze(0).to(self.device)

                modalities_used.append("audio")

        # ─── Process Video ───
        if video_features is not None:
            openface = video_features.get("openface")
            cnn_embed = video_features.get("cnn_embed")

            if openface is not None and cnn_embed is not None:
                openface = self._normalize_features(openface)
                cnn_embed = self._normalize_features(cnn_embed)

                min_t = min(openface.shape[0], cnn_embed.shape[0])
                openface = openface[:min_t]
                cnn_embed = cnn_embed[:min_t]

                kwargs["openface"] = self._to_tensor(openface)
                kwargs["cnn_embed"] = self._to_tensor(cnn_embed)
                kwargs["video_mask"] = self._make_mask(min_t)

                modalities_used.append("video")

        # ─── Process Text ───
        if text_features is not None:
            text = self._normalize_features(text_features)
            kwargs["text"] = self._to_tensor(text)
            kwargs["text_mask"] = self._make_mask(text.shape[0])
            modalities_used.append("text")

        if not modalities_used:
            raise ValueError("No valid modality features provided")

        # ─── Run Model ───
        output: FusionOutput = self.model(**kwargs)

        score = float(output.prediction.squeeze().clamp(0.0, 24.0).item())
        elapsed = time.perf_counter() - t_start

        # Compute confidence based on number of modalities
        confidence = len(modalities_used) / 3.0  # Simple heuristic: more modalities = more confident

        logger.info(
            f"Multimodal prediction: pid={participant_id}, score={score:.2f}, "
            f"modalities={modalities_used}, time={elapsed:.3f}s"
        )

        debug_info = None
        if debug:
            debug_info = {
                "gate_values": output.gate_values,
                "modalities_used": modalities_used,
                "model_params": self.model.param_summary(),
            }

        return MultimodalPredictionResult(
            phq8_score=round(score, 2),
            severity=MultimodalPredictionResult.severity_label(score),
            confidence=round(confidence, 2),
            modalities_used=modalities_used,
            modality_contributions={
                "audio": output.audio_contribution,
                "video": output.video_contribution,
                "text": output.text_contribution,
            },
            inference_time_s=round(elapsed, 3),
            participant_id=participant_id,
            debug=debug_info,
        )

    @torch.no_grad()
    def predict_from_files(
        self,
        mfcc_csv: Optional[str] = None,
        egemaps_csv: Optional[str] = None,
        behavioral_csv: Optional[str] = None,
        openface_csv: Optional[str] = None,
        cnn_csv: Optional[str] = None,
        text_csv: Optional[str] = None,
        participant_id: str = "unknown",
        debug: bool = False,
    ) -> MultimodalPredictionResult:
        """
        Predict from CSV feature files.

        Args:
            mfcc_csv: Path to MFCC features (N, 120)
            egemaps_csv: Path to eGeMAPS features (N, 88)
            behavioral_csv: Path to behavioral features (16,)
            openface_csv: Path to OpenFace features (T, 49)
            cnn_csv: Path to CNN embeddings (T, 512)
            text_csv: Path to text embeddings (N, 384)
        """
        audio_features = None
        video_features = None
        text_features = None

        # Load audio
        mfcc = self._load_csv_features(mfcc_csv) if mfcc_csv else None
        egemaps = self._load_csv_features(egemaps_csv) if egemaps_csv else None
        behavioral = self._load_csv_features(behavioral_csv) if behavioral_csv else None
        if behavioral is not None:
            behavioral = behavioral.flatten()[:16]

        if mfcc is not None and egemaps is not None:
            audio_features = {"mfcc": mfcc, "egemaps": egemaps}
            if behavioral is not None:
                audio_features["behavioral"] = behavioral

        # Load video
        openface = self._load_csv_features(openface_csv) if openface_csv else None
        cnn_embed = self._load_csv_features(cnn_csv) if cnn_csv else None
        if openface is not None and cnn_embed is not None:
            video_features = {"openface": openface, "cnn_embed": cnn_embed}

        # Load text
        if text_csv:
            text_features = self._load_csv_features(text_csv)

        return self.predict(
            audio_features=audio_features,
            video_features=video_features,
            text_features=text_features,
            participant_id=participant_id,
            debug=debug,
        )

    def predict_batch(
        self,
        sessions: List[Dict[str, Any]],
    ) -> List[MultimodalPredictionResult]:
        """
        Batch prediction for multiple sessions.

        Args:
            sessions: List of dicts, each with keys matching predict() arguments
        """
        results = []
        for session in sessions:
            try:
                result = self.predict(**session)
                results.append(result)
            except Exception as e:
                pid = session.get("participant_id", "unknown")
                logger.error(f"Failed for {pid}: {e}")
        return results
