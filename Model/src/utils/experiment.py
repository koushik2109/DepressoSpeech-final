from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    run_directory: str = "runs"


class ExperimentManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.root = Path(config.run_directory) / config.name
        self.root.mkdir(parents=True, exist_ok=True)
        self.seed(config.seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def save_config(self, filename: str = "config.json") -> Path:
        path = self.root / filename
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self.config), handle, indent=2)
        return path

    def checkpoint_path(self, name: str = "best_checkpoint.pt") -> Path:
        return self.root / name
