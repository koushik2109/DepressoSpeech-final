from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


class PCAReducer:
    def __init__(self, n_components: int = 64, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten
        self._pca: PCA | None = None

    def fit(self, data: np.ndarray) -> "PCAReducer":
        self._pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self._pca.fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise ValueError("PCA reducer must be fit before calling transform()")
        return self._pca.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"n_components": self.n_components, "whiten": self.whiten, "pca": self._pca}, handle)

    @classmethod
    def load(cls, path: Path) -> "PCAReducer":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        reducer = cls(n_components=payload["n_components"], whiten=payload["whiten"])
        reducer._pca = payload["pca"]
        return reducer
