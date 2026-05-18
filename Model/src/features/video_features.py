import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T

from src.features.sanitization import sanitize_array


class OpenFaceFeatureLoader:
    """Load OpenFace CSV features for consistent train and inference."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

    def load(self) -> Dict[str, np.ndarray]:
        with open(self.csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        if not rows:
            raise ValueError(f"OpenFace CSV is empty: {self.csv_path}")

        values = np.array([[float(v or 0.0) for v in row.values()] for row in rows], dtype=np.float32)
        columns = list(rows[0].keys())
        return {
            "features": sanitize_array(values),
            "columns": np.array(columns, dtype=object),
        }


class VideoEmbeddingExtractor:
    """Visual embedding backend for ResNet50 or placeholder VideoMAE."""

    def __init__(self, backend: str = "resnet50", device: str = "cpu"):
        self.backend = backend
        self.device = device
        if backend == "resnet50":
            self.model = models.resnet50(pretrained=True).to(device).eval()
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        else:
            raise ValueError(f"Unsupported video embedding backend: {backend}")
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        tensors = [self.transform(frame).to(self.device) for frame in frames]
        batch = torch.stack(tensors, dim=0)
        with torch.no_grad():
            embeddings = self.model(batch).squeeze(-1).squeeze(-1)
        return sanitize_array(embeddings.cpu().numpy())


class VideoFeatureExtractor:
    """Combine behavioral OpenFace features with visual embedding features."""

    def __init__(self, config: dict, device: str = "cpu"):
        self.loader_backend = config.get("openface_csv_key", "openface")
        self.embedding_backend = config.get("embedding_backend", "resnet50")
        self.embedding_extractor = VideoEmbeddingExtractor(self.embedding_backend, device=device)

    def extract_openface(self, csv_path: Path) -> Dict[str, np.ndarray]:
        loader = OpenFaceFeatureLoader(csv_path)
        loaded = loader.load()
        features = loaded["features"]
        return {
            "behavior": features,
            "columns": loaded["columns"],
        }

    def embed_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        return self.embedding_extractor.embed_frames(frames)
