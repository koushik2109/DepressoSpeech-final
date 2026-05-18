from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any
from torch.utils.data import Dataset

from src.features.feature_store import load_features
from src.features.sanitization import sanitize_array


class MultimodalSample(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        augment: bool = False,
        temporal_dropout_rate: float = 0.1,
        feature_noise_std: float = 0.01,
    ) -> None:
        self.samples = samples
        self.augment = augment
        self.temporal_dropout_rate = temporal_dropout_rate
        self.feature_noise_std = feature_noise_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        audio = self._prepare_feature(sample.get("audio"))
        video = self._prepare_feature(sample.get("video"))
        text = self._prepare_feature(sample.get("text"))

        if self.augment:
            audio, video, text = self._augment(audio, video, text)

        return {
            "participant_id": sample["participant_id"],
            "audio": torch.from_numpy(audio) if audio is not None else None,
            "video": torch.from_numpy(video) if video is not None else None,
            "text": torch.from_numpy(text) if text is not None else None,
            "audio_mask": torch.ones(audio.shape[0], dtype=torch.bool) if audio is not None else torch.zeros(0, dtype=torch.bool),
            "video_mask": torch.ones(video.shape[0], dtype=torch.bool) if video is not None else torch.zeros(0, dtype=torch.bool),
            "text_mask": torch.ones(text.shape[0], dtype=torch.bool) if text is not None else torch.zeros(0, dtype=torch.bool),
            "phq_total": torch.tensor(sample["phq_total"], dtype=torch.float32),
            "phq_questions": torch.tensor(sample["phq_questions"], dtype=torch.float32),
            "classification": torch.tensor(sample["classification"], dtype=torch.float32),
        }

    @staticmethod
    def _prepare_feature(feature: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if feature is None:
            return None
        feature = sanitize_array(feature)
        if feature.ndim == 1:
            raise ValueError("Feature arrays must be 2D with shape [time, dim], but received 1D input.")
        if feature.ndim != 2:
            raise ValueError("Feature arrays must be 2D with shape [time, dim].")
        return feature

    @staticmethod
    def _temporal_dropout(feat: np.ndarray, rate: float) -> np.ndarray:
        """Drop random time steps from a feature sequence."""
        if rate <= 0.0 or feat is None:
            return feat
        keep = np.random.rand(feat.shape[0]) > rate
        if keep.sum() < 2:
            return feat
        return feat[keep]

    @staticmethod
    def _freq_mask(feat: np.ndarray, max_mask_frac: float = 0.20) -> np.ndarray:
        """SpecAugment-style: zero out a contiguous block of feature dimensions."""
        if feat is None or feat.shape[1] < 4:
            return feat
        n_dims = feat.shape[1]
        mask_width = max(1, int(np.random.uniform(0, max_mask_frac) * n_dims))
        start = np.random.randint(0, n_dims - mask_width + 1)
        feat = feat.copy()
        feat[:, start: start + mask_width] = 0.0
        return feat

    @staticmethod
    def _time_warp(feat: np.ndarray, warp_frac: float = 0.10) -> np.ndarray:
        """Randomly sub-sample or repeat time steps to simulate speed variation."""
        if feat is None or feat.shape[0] < 4:
            return feat
        T = feat.shape[0]
        new_T = max(2, int(T * np.random.uniform(1.0 - warp_frac, 1.0 + warp_frac)))
        indices = np.round(np.linspace(0, T - 1, new_T)).astype(int).clip(0, T - 1)
        return feat[indices]

    def _augment(
        self,
        audio: Optional[np.ndarray],
        video: Optional[np.ndarray],
        text: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        # Temporal dropout (random frame removal)
        audio = self._temporal_dropout(audio, self.temporal_dropout_rate)
        video = self._temporal_dropout(video, self.temporal_dropout_rate)
        text  = self._temporal_dropout(text,  self.temporal_dropout_rate)

        # SpecAugment-style feature dimension masking
        if np.random.rand() < 0.5:
            audio = self._freq_mask(audio, max_mask_frac=0.20)
        if np.random.rand() < 0.5:
            video = self._freq_mask(video, max_mask_frac=0.20)
        if np.random.rand() < 0.4:
            text  = self._freq_mask(text,  max_mask_frac=0.15)

        # Time-warp jitter (speed perturbation)
        if np.random.rand() < 0.5:
            audio = self._time_warp(audio, warp_frac=0.10)
        if np.random.rand() < 0.5:
            video = self._time_warp(video, warp_frac=0.10)

        # Additive Gaussian noise
        if self.feature_noise_std > 0.0:
            if audio is not None:
                audio = audio + np.random.normal(0, self.feature_noise_std, audio.shape).astype(np.float32)
            if video is not None:
                video = video + np.random.normal(0, self.feature_noise_std, video.shape).astype(np.float32)
            if text is not None:
                text = text + np.random.normal(0, self.feature_noise_std * 0.5, text.shape).astype(np.float32)
        return audio, video, text

    def feature_dimensions(self) -> Dict[str, int]:
        dims: Dict[str, int] = {"audio": 0, "video": 0, "text": 0}
        for sample in self.samples:
            for key in dims:
                feature = sample.get(key)
                if feature is not None and getattr(feature, "ndim", 0) == 2:
                    dims[key] = feature.shape[1]
                    if dims[key] > 0:
                        break
        return dims

    @classmethod
    def from_feature_store(
        cls,
        split_csv_path: Path,
        feature_dir: Path,
        label_column: str = "phq_total",
        id_column: str = "participant_id",
        modality_keys: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> "MultimodalSample":
        import pandas as pd

        df = pd.read_csv(split_csv_path)
        samples = []
        modality_keys = modality_keys or ["audio", "video", "text"]
        for _, row in df.iterrows():
            participant_id = str(row[id_column])
            feature_path = feature_dir / f"{participant_id}.npz"
            if not feature_path.exists():
                continue
            features = load_features(feature_path)
            sample = {
                "participant_id": participant_id,
                "audio": features.get("audio") if "audio" in modality_keys else None,
                "video": features.get("video") if "video" in modality_keys else None,
                "text": features.get("text") if "text" in modality_keys else None,
                "phq_total": float(row[label_column]),
                "phq_questions": np.asarray(
                    [row.get(f"phq_q{i}", 0.0) for i in range(1, 9)],
                    dtype=np.float32,
                ),
                "classification": float(row.get("binary", float(row[label_column] >= 10))),
            }
            samples.append(sample)
        return cls(samples=samples, **kwargs)
