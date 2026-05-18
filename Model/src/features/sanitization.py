import numpy as np


def sanitize_array(array: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=np.float32)
    else:
        array = array.astype(np.float32, copy=False)
    array = np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return array
