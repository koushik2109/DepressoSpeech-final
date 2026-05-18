"""Robust feature normalization and PCA transformation with serialization.

This module provides standardized normalization and PCA transforms that ensure
identical preprocessing between training and inference.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


class NormalizerRegistry:
    """Registry of supported normalization methods."""

    _registry = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }

    @classmethod
    def get(cls, name: str) -> type:
        """Get normalizer class by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown normalizer: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def register(cls, name: str, normalizer_class: type) -> None:
        """Register a custom normalizer."""
        cls._registry[name] = normalizer_class


class FeatureNormalizer:
    """Fit and apply feature normalization with serialization."""

    def __init__(self, normalizer_type: str = "standard", epsilon: float = 1e-8):
        """Initialize normalizer.

        Args:
            normalizer_type: Type of normalizer ("standard", "robust", "minmax")
            epsilon: Small value to prevent division by zero
        """
        self.normalizer_type = normalizer_type
        self.epsilon = epsilon
        normalizer_class = NormalizerRegistry.get(normalizer_type)
        self._normalizer = normalizer_class()
        self._fitted = False

    def fit(self, features: np.ndarray) -> FeatureNormalizer:
        """Fit normalizer to features."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 3:  # Batch of temporal features [batch, time, dim]
            features = features.reshape(-1, features.shape[-1])
        elif features.ndim == 2:  # Single or multiple [time, dim]
            pass
        else:
            raise ValueError(f"Expected 2D or 3D features, got {features.ndim}D")

        self._normalizer.fit(features)
        self._fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if not self._fitted:
            raise ValueError("Normalizer must be fit before calling transform()")

        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 2:
            return self._normalizer.transform(features).astype(np.float32)
        elif features.ndim == 3:
            batch_size, time_steps, dim = features.shape
            reshaped = features.reshape(-1, dim)
            transformed = self._normalizer.transform(reshaped)
            return transformed.reshape(batch_size, time_steps, dim).astype(np.float32)
        else:
            raise ValueError(f"Expected 2D or 3D features, got {features.ndim}D")

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform features."""
        self.fit(features)
        return self.transform(features)

    def save(self, path: Path) -> None:
        """Save normalizer to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "normalizer_type": self.normalizer_type,
                "epsilon": self.epsilon,
                "normalizer": self._normalizer,
                "fitted": self._fitted,
            }, f)

    @classmethod
    def load(cls, path: Path) -> FeatureNormalizer:
        """Load normalizer from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        
        normalizer = cls(
            normalizer_type=payload["normalizer_type"],
            epsilon=payload["epsilon"],
        )
        normalizer._normalizer = payload["normalizer"]
        normalizer._fitted = payload["fitted"]
        return normalizer


class PCATransform:
    """Fit and apply PCA transformation with serialization."""

    def __init__(self, n_components: int | float = 0.95, whiten: bool = False):
        """Initialize PCA transform.

        Args:
            n_components: Number of principal components to keep, or float for variance ratio
            whiten: Whether to apply whitening
        """
        self.n_components = n_components
        self.whiten = whiten
        self._pca: Optional[PCA] = None
        self._fitted = False

    def fit(self, features: np.ndarray) -> PCATransform:
        """Fit PCA to features."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 3:  # Batch of temporal features [batch, time, dim]
            features = features.reshape(-1, features.shape[-1])
        elif features.ndim == 2:
            pass
        else:
            raise ValueError(f"Expected 2D or 3D features, got {features.ndim}D")

        self._pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self._pca.fit(features)
        self._fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply PCA transformation."""
        if not self._fitted or self._pca is None:
            raise ValueError("PCA must be fit before calling transform()")

        features = np.asarray(features, dtype=np.float32)
        original_shape = None
        
        if features.ndim == 3:  # Batch of temporal features
            batch_size, time_steps, dim = features.shape
            original_shape = (batch_size, time_steps)
            features = features.reshape(-1, dim)
        elif features.ndim == 2:
            pass
        else:
            raise ValueError(f"Expected 2D or 3D features, got {features.ndim}D")

        transformed = self._pca.transform(features).astype(np.float32)
        
        if original_shape is not None:
            transformed = transformed.reshape(original_shape[0], original_shape[1], -1)
        
        return transformed

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform features."""
        self.fit(features)
        return self.transform(features)

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio."""
        return self._pca.explained_variance_ratio_ if self._fitted and self._pca else None

    @property
    def total_variance_explained(self) -> Optional[float]:
        """Get total variance explained."""
        if self.explained_variance_ratio is not None:
            return float(self.explained_variance_ratio.sum())
        return None

    def save(self, path: Path) -> None:
        """Save PCA transform to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "whiten": self.whiten,
                "pca": self._pca,
                "fitted": self._fitted,
            }, f)

    @classmethod
    def load(cls, path: Path) -> PCATransform:
        """Load PCA transform from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        
        transform = cls(
            n_components=payload["n_components"],
            whiten=payload["whiten"],
        )
        transform._pca = payload["pca"]
        transform._fitted = payload["fitted"]
        return transform


class FeaturePreprocessor:
    """Unified feature preprocessing: normalization + optional PCA."""

    def __init__(
        self,
        normalizer_type: str = "standard",
        apply_pca: bool = False,
        pca_components: int | float = 0.95,
        pca_whiten: bool = False,
    ):
        """Initialize preprocessor.

        Args:
            normalizer_type: Type of normalization ("standard", "robust", "minmax")
            apply_pca: Whether to apply PCA
            pca_components: Number of PCA components if apply_pca=True
            pca_whiten: Whether to apply whitening in PCA
        """
        self.normalizer_type = normalizer_type
        self.apply_pca = apply_pca
        self.pca_components = pca_components
        self.pca_whiten = pca_whiten

        self.normalizer = FeatureNormalizer(normalizer_type=normalizer_type)
        self.pca: Optional[PCATransform] = None
        if apply_pca:
            self.pca = PCATransform(n_components=pca_components, whiten=pca_whiten)

    def fit(self, features: np.ndarray) -> FeaturePreprocessor:
        """Fit normalizer and PCA."""
        self.normalizer.fit(features)
        if self.apply_pca and self.pca:
            normalized = self.normalizer.transform(features)
            self.pca.fit(normalized)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply normalization and PCA."""
        normalized = self.normalizer.transform(features)
        if self.apply_pca and self.pca:
            return self.pca.transform(normalized)
        return normalized

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform features."""
        self.fit(features)
        return self.transform(features)

    @property
    def output_dim(self) -> int:
        """Get output feature dimension."""
        if self.apply_pca and self.pca and self.pca._fitted:
            return self.pca._pca.n_components_
        return None  # Unknown without fitting

    def save(self, path: Path) -> None:
        """Save preprocessor to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "normalizer_type": self.normalizer_type,
                "apply_pca": self.apply_pca,
                "pca_components": self.pca_components,
                "pca_whiten": self.pca_whiten,
                "normalizer": self.normalizer,
                "pca": self.pca,
            }, f)

    @classmethod
    def load(cls, path: Path) -> FeaturePreprocessor:
        """Load preprocessor from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        
        preprocessor = cls(
            normalizer_type=payload["normalizer_type"],
            apply_pca=payload["apply_pca"],
            pca_components=payload["pca_components"],
            pca_whiten=payload["pca_whiten"],
        )
        preprocessor.normalizer = payload["normalizer"]
        preprocessor.pca = payload["pca"]
        return preprocessor
