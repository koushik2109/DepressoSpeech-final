import numpy as np


def simple_energy_vad(waveform: np.ndarray, sr: int, energy_threshold: float = 0.02) -> bool:
    if waveform.size == 0:
        return False
    energy = np.mean(waveform ** 2)
    return energy > energy_threshold
