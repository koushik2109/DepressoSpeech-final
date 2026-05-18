from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.features.feature_store import load_features
from src.dataset.multimodal_dataset import MultimodalSample


class DatasetBuilder:
    @staticmethod
    def build_from_csv(
        split_csv_path: Path,
        feature_dir: Path,
        id_column: str = "participant_id",
        label_column: str = "phq_total",
        modality_keys: Optional[List[str]] = None,
        augment: bool = False,
        **kwargs,
    ) -> MultimodalSample:
        df = pd.read_csv(split_csv_path)
        df.columns = df.columns.str.lower().str.strip()
        id_column = id_column.lower()
        label_column = label_column.lower()
        samples = []
        modality_keys = modality_keys or ["audio", "video", "text"]
        for _, row in df.iterrows():
            participant_id = str(row[id_column])
            feature_path = feature_dir / f"{participant_id}.npz"
            if not feature_path.exists():
                continue
            features = load_features(feature_path)
            phq_questions = np.asarray(
                [row.get(f"phq_q{i}", 0.0) for i in range(1, 9)],
                dtype=np.float32,
            )
            classification = row.get("binary")
            if classification is None:
                classification = row.get("phq_binary", float(row[label_column] >= 10))

            video_feature = None
            if "video" in modality_keys:
                if "video" in features:
                    video_feature = features["video"]
                elif "video_openface" in features:
                    video_feature = features["video_openface"]
                elif "video_cnn" in features:
                    video_feature = features["video_cnn"]

            sample = {
                "participant_id": participant_id,
                "audio": features.get("audio") if "audio" in modality_keys else None,
                "video": video_feature,
                "text": features.get("text") if "text" in modality_keys else None,
                "phq_total": float(row[label_column]),
                "phq_questions": phq_questions,
                "classification": float(classification),
            }
            samples.append(sample)
        return MultimodalSample(samples=samples, augment=augment, **kwargs)
