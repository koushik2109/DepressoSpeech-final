"""Feature specification and validation system for training/inference consistency.

This module defines strict feature specifications for each modality to ensure
training and inference produce identical feature representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class ModalitySpec:
    """Specification for a single modality's features."""

    name: str
    expected_dim: int
    temporal: bool = True
    preprocessor_class: Optional[str] = None
    extractor_class: Optional[str] = None
    pca_enabled: bool = False
    pca_target_dim: Optional[int] = None
    normalizer_type: str = "standard"  # "standard", "robust", "minmax", "none"

    def validate_features(self, features: np.ndarray) -> bool:
        """Validate that features match specification."""
        if not isinstance(features, np.ndarray):
            return False
        if features.dtype != np.float32:
            return False
        if self.temporal:
            if features.ndim != 2:
                return False
            if features.shape[1] != self.expected_dim:
                return False
        else:
            if features.ndim != 1:
                return False
            if features.shape[0] != self.expected_dim:
                return False
        return not (np.isnan(features).any() or np.isinf(features).any())

    def to_dict(self) -> Dict:
        """Serialize spec to dict."""
        return {
            "name": self.name,
            "expected_dim": self.expected_dim,
            "temporal": self.temporal,
            "preprocessor_class": self.preprocessor_class,
            "extractor_class": self.extractor_class,
            "pca_enabled": self.pca_enabled,
            "pca_target_dim": self.pca_target_dim,
            "normalizer_type": self.normalizer_type,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> ModalitySpec:
        """Deserialize spec from dict."""
        return cls(**data)


class AudioSpec:
    """Audio modality specifications."""

    # MFCC-based audio: 13 MFCC + 13 delta + 13 delta2 = 39
    MFCC_DIM = 39

    # HuBERT-base embeddings
    HUBERT_DIM = 768

    # Combined features: MFCC (39) + HuBERT (768) = 807
    COMBINED_DIM = MFCC_DIM + HUBERT_DIM

    # Standard specification: use MFCC features for training
    STANDARD = ModalitySpec(
        name="audio",
        expected_dim=MFCC_DIM,  # 39: 13 MFCC + 13 delta + 13 delta2
        temporal=True,
        preprocessor_class="AudioPreprocessor",
        extractor_class="AudioFeatureExtractor",
        pca_enabled=False,
        normalizer_type="standard",
    )

    # Optional: HuBERT embeddings
    HUBERT = ModalitySpec(
        name="audio_hubert",
        expected_dim=HUBERT_DIM,  # 768
        temporal=True,
        preprocessor_class="AudioPreprocessor",
        extractor_class="AudioFeatureExtractor",
        pca_enabled=True,
        pca_target_dim=128,
        normalizer_type="standard",
    )

    # Optional: Combined MFCC + HuBERT
    COMBINED = ModalitySpec(
        name="audio_combined",
        expected_dim=COMBINED_DIM,  # 807
        temporal=True,
        preprocessor_class="AudioPreprocessor",
        extractor_class="AudioFeatureExtractor",
        pca_enabled=True,
        pca_target_dim=128,
        normalizer_type="standard",
    )


class VideoSpec:
    """Video modality specifications."""

    # OpenFace features: AU (17) + gaze (3) + head pose (3) + landmarks dynamics (many)
    OPENFACE_DIM = 38  # Conservative estimate for behavioral features

    # ResNet50 embeddings: 2048D, typically PCA-reduced
    RESNET_RAW_DIM = 2048
    RESNET_PCA_DIM = 128  # Standard reduction for ResNet

    # Specification: behavioral features + visual embeddings
    BEHAVIORAL = ModalitySpec(
        name="video_behavior",
        expected_dim=OPENFACE_DIM,  # 38: OpenFace AU + gaze + pose
        temporal=True,
        preprocessor_class="VideoPreprocessor",
        extractor_class="VideoFeatureExtractor",
        pca_enabled=False,
        normalizer_type="standard",
    )

    VISUAL_EMBEDDING = ModalitySpec(
        name="video_embedding",
        expected_dim=RESNET_PCA_DIM,  # 128: ResNet50 PCA-reduced
        temporal=True,
        preprocessor_class="VideoPreprocessor",
        extractor_class="VideoFeatureExtractor",
        pca_enabled=True,  # PCA is applied during extraction
        pca_target_dim=RESNET_PCA_DIM,
        normalizer_type="standard",
    )

    # Combined: behavioral + visual
    COMBINED = ModalitySpec(
        name="video_combined",
        expected_dim=OPENFACE_DIM + RESNET_PCA_DIM,  # 166: behavioral + visual
        temporal=True,
        preprocessor_class="VideoPreprocessor",
        extractor_class="VideoFeatureExtractor",
        pca_enabled=False,
        normalizer_type="standard",
    )

    # Standard: behavioral features only (recommended for small datasets)
    STANDARD = BEHAVIORAL


class TextSpec:
    """Text modality specifications."""

    # SBERT all-mpnet-base-v2 embeddings
    SBERT_DIM = 384

    # Standard specification
    STANDARD = ModalitySpec(
        name="text",
        expected_dim=SBERT_DIM,  # 384: SBERT embeddings
        temporal=True,  # Multiple chunks per session
        preprocessor_class="TextPreprocessor",
        extractor_class="TextFeatureExtractor",
        pca_enabled=False,
        normalizer_type="standard",
    )


class FeatureSpecification:
    """Complete multimodal feature specification for training/inference consistency."""

    def __init__(
        self,
        audio_spec: ModalitySpec = AudioSpec.STANDARD,
        video_spec: ModalitySpec = VideoSpec.STANDARD,
        text_spec: ModalitySpec = TextSpec.STANDARD,
    ):
        self.audio_spec = audio_spec
        self.video_spec = video_spec
        self.text_spec = text_spec
        self._specs = {
            "audio": audio_spec,
            "video": video_spec,
            "text": text_spec,
        }

    @property
    def audio_dim(self) -> int:
        return self.audio_spec.expected_dim

    @property
    def video_dim(self) -> int:
        return self.video_spec.expected_dim

    @property
    def text_dim(self) -> int:
        return self.text_spec.expected_dim

    def get_spec(self, modality: str) -> ModalitySpec:
        """Get specification for a modality."""
        if modality not in self._specs:
            raise ValueError(f"Unknown modality: {modality}")
        return self._specs[modality]

    def validate(self, features: Dict[str, np.ndarray]) -> bool:
        """Validate all feature modalities match specification."""
        for modality, feature_array in features.items():
            if modality in self._specs:
                if not self._specs[modality].validate_features(feature_array):
                    return False
        return True

    def to_dict(self) -> Dict:
        """Serialize specification."""
        return {
            "audio": self.audio_spec.to_dict(),
            "video": self.video_spec.to_dict(),
            "text": self.text_spec.to_dict(),
        }

    @classmethod
    def default(cls) -> FeatureSpecification:
        """Create default specification: MFCC audio + behavioral video + SBERT text."""
        return cls(
            audio_spec=AudioSpec.STANDARD,
            video_spec=VideoSpec.STANDARD,
            text_spec=TextSpec.STANDARD,
        )

    @classmethod
    def with_hubert(cls) -> FeatureSpecification:
        """Create specification with HuBERT audio embeddings."""
        return cls(
            audio_spec=AudioSpec.HUBERT,
            video_spec=VideoSpec.STANDARD,
            text_spec=TextSpec.STANDARD,
        )

    @classmethod
    def multimodal_combined(cls) -> FeatureSpecification:
        """Create specification with all combined features."""
        return cls(
            audio_spec=AudioSpec.COMBINED,
            video_spec=VideoSpec.COMBINED,
            text_spec=TextSpec.STANDARD,
        )
