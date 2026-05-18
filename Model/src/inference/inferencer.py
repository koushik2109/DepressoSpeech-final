from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.inference.model_io import load_checkpoint, sanitize_payload
from src.models.multimodal_model import MultimodalDepressionModel
from src.dataset.collate import multimodal_collate_fn


class ModelV2Inferencer:
    def __init__(self, checkpoint_path: Path, model_config: Dict[str, Any], device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = MultimodalDepressionModel(**model_config)
        load_checkpoint(self.model, checkpoint_path, device=str(self.device))
        self.model.to(self.device).eval()

    def _prepare_mask(self, values: Any, mask: Any) -> torch.Tensor:
        if mask is not None:
            tensor_mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
            if tensor_mask.dim() == 1:
                tensor_mask = tensor_mask.unsqueeze(0)
            return tensor_mask

        if values is not None:
            length = values.shape[0]
            return torch.ones((1, length), dtype=torch.bool, device=self.device)

        return torch.zeros((1, 1), dtype=torch.bool, device=self.device)

    def predict_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = sanitize_payload(sample)
        with torch.no_grad():
            audio = sample.get("audio")
            video = sample.get("video")
            text = sample.get("text")
            if audio is not None:
                audio = torch.tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0)
            if video is not None:
                video = torch.tensor(video, dtype=torch.float32, device=self.device).unsqueeze(0)
            if text is not None:
                text = torch.tensor(text, dtype=torch.float32, device=self.device).unsqueeze(0)

            audio_mask = self._prepare_mask(audio, sample.get("audio_mask"))
            video_mask = self._prepare_mask(video, sample.get("video_mask"))
            text_mask = self._prepare_mask(text, sample.get("text_mask"))

            output = self.model(audio, video, text, audio_mask, video_mask, text_mask)
            return {
                "phq_total": float(output["phq_total"].cpu().numpy().mean()),
                "phq_questions": output["phq_questions"].cpu().numpy().flatten().tolist(),
                "classification": float(torch.sigmoid(output["classification"]).cpu().numpy().mean()),
                "confidence": float(output["confidence"].cpu().numpy().mean()),
                "modality_scores": {k: float(v.cpu().numpy().mean()) for k, v in output["modality_scores"].items()},
                "entropy": float(output["entropy"].cpu().numpy().mean()),
            }

    def predict_batch(self, batch: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        results = []
        for sample in batch:
            results.append(self.predict_single(sample))
        return results
