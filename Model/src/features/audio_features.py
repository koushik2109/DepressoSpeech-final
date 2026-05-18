import math
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src.features.sanitization import sanitize_array


class AudioFeatureExtractor:
    """Audio feature extraction for train and inference.

    Supports classical features (MFCC, delta, silence ratio) and
    modern self-supervised embeddings (HuBERT/Wav2Vec2).
    """

    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 16000)
        self.mfcc_n_mfcc = config.get("mfcc_n_mfcc", 13)
        self.hubert_model_name = config.get("hubert_model_name", "facebook/hubert-base-ls960")
        self.feature_backends = tuple(config.get("feature_backends", ["mfcc", "hubert"]))

        self._hubert_extractor: Optional[Wav2Vec2FeatureExtractor] = None
        self._hubert_model: Optional[Wav2Vec2Model] = None

    def _ensure_hubert(self):
        if self._hubert_model is None:
            self._hubert_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.hubert_model_name)
            self._hubert_model = Wav2Vec2Model.from_pretrained(self.hubert_model_name)
            self._hubert_model.eval()

    def extract_classical(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        waveform = waveform.astype(np.float32)
        waveform = librosa.util.normalize(waveform)
        mfcc = librosa.feature.mfcc(y=waveform, sr=self.sample_rate, n_mfcc=self.mfcc_n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        energy = librosa.feature.rms(y=waveform)[0]
        silence_ratio = float(np.mean(energy < np.percentile(energy, 10)))
        speech_rate = float(np.sum(energy > np.median(energy)) / max(1, len(energy)))

        return {
            "mfcc": sanitize_array(mfcc.T),
            "delta": sanitize_array(delta.T),
            "delta2": sanitize_array(delta2.T),
            "silence_ratio": np.array([silence_ratio], dtype=np.float32),
            "speech_rate": np.array([speech_rate], dtype=np.float32),
        }

    def extract_modern(self, waveform: np.ndarray) -> np.ndarray:
        self._ensure_hubert()
        waveform = waveform.astype(np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        inputs = self._hubert_extractor(waveform, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._hubert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return sanitize_array(embedding.squeeze(0).cpu().numpy())

    def extract_features(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        features: Dict[str, np.ndarray] = {}
        if "mfcc" in self.feature_backends:
            classical = self.extract_classical(waveform)
            features.update(classical)
        if "hubert" in self.feature_backends:
            features["hubert"] = self.extract_modern(waveform)
        return features
