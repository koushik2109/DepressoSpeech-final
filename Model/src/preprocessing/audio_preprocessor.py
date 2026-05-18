import asyncio
import math
import numpy as np
import librosa
from pathlib import Path
from typing import Sequence

from src.preprocessing.vad import simple_energy_vad
from src.preprocessing.chunker import sliding_window
from src.features.sanitization import sanitize_array


class AudioPreprocessor:
    """Preprocessing pipeline for raw audio waveforms."""

    def __init__(self, sample_rate: int = 16000, frame_rate: int = 8, chunk_duration: float = 2.0, chunk_overlap: float = 0.5):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap

    async def load_audio(self, path: Path) -> np.ndarray:
        waveform, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return sanitize_array(waveform)

    def extract_chunks(self, waveform: np.ndarray) -> Sequence[np.ndarray]:
        step = self.chunk_duration - self.chunk_overlap
        hop_length = int(step * self.sample_rate)
        window_length = int(self.chunk_duration * self.sample_rate)
        return [segment for segment in sliding_window(waveform, window_length, hop_length) if segment.size == window_length]

    def apply_vad(self, chunks: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        return [chunk for chunk in chunks if simple_energy_vad(chunk, self.sample_rate)]

    def preprocess(self, waveform: np.ndarray) -> np.ndarray:
        waveform = sanitize_array(waveform)
        chunks = self.extract_chunks(waveform)
        chunks = self.apply_vad(chunks)
        if not chunks:
            return np.zeros((1, int(self.chunk_duration * self.sample_rate)), dtype=np.float32)
        return np.stack(chunks, axis=0)
