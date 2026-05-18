from typing import Sequence
import numpy as np


def sliding_window(array: np.ndarray, window_size: int, step_size: int) -> Sequence[np.ndarray]:
    return [array[i : i + window_size] for i in range(0, len(array) - window_size + 1, step_size)]
