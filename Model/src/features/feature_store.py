import numpy as np
from pathlib import Path
from typing import Any, Dict


def save_features(path: Path, features: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)


def load_features(path: Path) -> Dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}
