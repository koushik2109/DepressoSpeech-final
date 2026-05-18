import asyncio
import subprocess
from pathlib import Path
from typing import List

import numpy as np


class VideoPreprocessor:
    """Async frame extraction and temporal window generation for video."""

    def __init__(self, frame_rate: int = 8, temporal_window: int = 32):
        self.frame_rate = frame_rate
        self.temporal_window = temporal_window

    async def extract_frames(self, video_path: Path) -> List[np.ndarray]:
        output_dir = Path(video_path.parent) / f"{video_path.stem}_frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={self.frame_rate}",
            str(output_dir / "%06d.jpg"),
        ]
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await process.communicate()
        paths = sorted(output_dir.glob("*.jpg"))
        frames = [np.array(__import__("PIL.Image").Image.open(str(p)).convert("RGB")) for p in paths]
        return frames

    def temporal_windows(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError("Video features must be a 2D array [frames, dim]")
        num_frames = features.shape[0]
        if num_frames == 0:
            return np.zeros((1, self.temporal_window, features.shape[1]), dtype=np.float32)
        stride = max(1, num_frames // self.temporal_window)
        indices = list(range(0, num_frames, stride))[: self.temporal_window]
        selected = features[indices]
        if selected.shape[0] < self.temporal_window:
            pad = np.zeros((self.temporal_window - selected.shape[0], features.shape[1]), dtype=np.float32)
            selected = np.concatenate([selected, pad], axis=0)
        return selected
