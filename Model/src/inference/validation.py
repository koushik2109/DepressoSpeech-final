"""Feature validation and train/inference consistency checking.

This module ensures that training and inference pipelines produce identical
feature representations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass

from src.features.specs import FeatureSpecification, ModalitySpec
from src.features.preprocessing import FeaturePreprocessor, PCATransform, FeatureNormalizer


@dataclass
class FeatureValidationResult:
    """Result of feature validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    feature_dims: Dict[str, int]

    def report(self) -> str:
        """Generate validation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("FEATURE VALIDATION REPORT")
        lines.append("=" * 60)
        
        if self.is_valid:
            lines.append("✓ All validations passed")
        else:
            lines.append("✗ Validation failed")
            for error in self.errors:
                lines.append(f"  ERROR: {error}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  WARNING: {warning}")
        
        lines.append("\nFeature Dimensions:")
        for modality, dim in self.feature_dims.items():
            lines.append(f"  {modality}: {dim}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class FeatureValidator:
    """Validate features match specification."""

    def __init__(self, spec: FeatureSpecification):
        """Initialize validator.

        Args:
            spec: Feature specification to validate against
        """
        self.spec = spec

    def validate_single_modality(
        self,
        features: np.ndarray,
        modality: str,
    ) -> Tuple[bool, List[str], List[str]]:
        """Validate single modality features.

        Args:
            features: Feature array
            modality: Modality name ("audio", "video", "text")

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        try:
            modal_spec = self.spec.get_spec(modality)
        except ValueError:
            errors.append(f"Unknown modality: {modality}")
            return False, errors, warnings

        # Check dtype
        if features.dtype != np.float32:
            errors.append(f"{modality}: dtype is {features.dtype}, expected float32")

        # Check dimensionality
        if modal_spec.temporal:
            if features.ndim != 2:
                errors.append(f"{modality}: expected 2D array [time, dim], got {features.ndim}D")
            else:
                if features.shape[1] != modal_spec.expected_dim:
                    errors.append(
                        f"{modality}: dimension mismatch. "
                        f"Expected {modal_spec.expected_dim}, got {features.shape[1]}"
                    )
                if features.shape[0] == 0:
                    errors.append(f"{modality}: zero-length temporal dimension")
        else:
            if features.ndim != 1:
                errors.append(f"{modality}: expected 1D array, got {features.ndim}D")
            if features.shape[0] != modal_spec.expected_dim:
                errors.append(
                    f"{modality}: dimension mismatch. "
                    f"Expected {modal_spec.expected_dim}, got {features.shape[0]}"
                )

        # Check for NaN/Inf
        if np.isnan(features).any():
            errors.append(f"{modality}: contains NaN values")
        if np.isinf(features).any():
            errors.append(f"{modality}: contains Inf values")

        # Warnings for extreme values
        if np.abs(features).max() > 100:
            warnings.append(f"{modality}: contains very large values (max={np.abs(features).max():.2f})")
        if np.abs(features[features != 0]).min() < 1e-6:
            warnings.append(f"{modality}: contains very small non-zero values")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def validate_batch(
        self,
        features: Dict[str, np.ndarray],
    ) -> FeatureValidationResult:
        """Validate complete batch of features.

        Args:
            features: Dict with "audio", "video", "text" keys

        Returns:
            FeatureValidationResult with detailed report
        """
        all_errors = []
        all_warnings = []
        feature_dims = {}

        for modality in ["audio", "video", "text"]:
            if modality in features and features[modality] is not None:
                array = features[modality]
                is_valid, errors, warnings = self.validate_single_modality(array, modality)
                all_errors.extend(errors)
                all_warnings.extend(warnings)
                
                if array.ndim == 2:
                    feature_dims[modality] = array.shape[1]
                elif array.ndim == 1:
                    feature_dims[modality] = array.shape[0]

        is_valid = len(all_errors) == 0
        return FeatureValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            feature_dims=feature_dims,
        )


class TrainInferenceConsistencyChecker:
    """Check that training and inference use consistent features."""

    def __init__(
        self,
        spec: FeatureSpecification,
        train_preprocessors: Optional[Dict[str, FeaturePreprocessor]] = None,
        inference_preprocessors: Optional[Dict[str, FeaturePreprocessor]] = None,
    ):
        """Initialize consistency checker.

        Args:
            spec: Feature specification
            train_preprocessors: Preprocessors used during training
            inference_preprocessors: Preprocessors used during inference
        """
        self.spec = spec
        self.train_preprocessors = train_preprocessors or {}
        self.inference_preprocessors = inference_preprocessors or {}
        self.validator = FeatureValidator(spec)

    def check_preprocessor_compatibility(self) -> Tuple[bool, List[str]]:
        """Check that training and inference preprocessors are compatible.

        Returns:
            Tuple of (is_compatible, issues)
        """
        issues = []

        for modality in ["audio", "video", "text"]:
            train_prep = self.train_preprocessors.get(modality)
            infer_prep = self.inference_preprocessors.get(modality)

            # Both should exist if either exists
            if (train_prep is None) != (infer_prep is None):
                issues.append(
                    f"{modality}: preprocessor exists in training but not in "
                    f"inference (or vice versa)"
                )
                continue

            # If both exist, check compatibility
            if train_prep and infer_prep:
                if train_prep.normalizer_type != infer_prep.normalizer_type:
                    issues.append(
                        f"{modality}: different normalizer types "
                        f"(train={train_prep.normalizer_type}, "
                        f"infer={infer_prep.normalizer_type})"
                    )

                if train_prep.apply_pca != infer_prep.apply_pca:
                    issues.append(
                        f"{modality}: PCA enabled in training but not in inference "
                        f"(or vice versa)"
                    )

                if train_prep.apply_pca and infer_prep.apply_pca:
                    if train_prep.pca_components != infer_prep.pca_components:
                        issues.append(
                            f"{modality}: different PCA components "
                            f"(train={train_prep.pca_components}, "
                            f"infer={infer_prep.pca_components})"
                        )

        is_compatible = len(issues) == 0
        return is_compatible, issues

    def validate_training_batch(self, batch: Dict[str, np.ndarray]) -> FeatureValidationResult:
        """Validate training batch."""
        return self.validator.validate_batch(batch)

    def validate_inference_batch(self, batch: Dict[str, np.ndarray]) -> FeatureValidationResult:
        """Validate inference batch."""
        return self.validator.validate_batch(batch)

    def generate_report(self) -> str:
        """Generate comprehensive consistency report."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("TRAIN/INFERENCE CONSISTENCY CHECK")
        lines.append("=" * 80)

        # Check preprocessor compatibility
        is_compatible, issues = self.check_preprocessor_compatibility()
        if is_compatible:
            lines.append("\n✓ Preprocessors are compatible between training and inference")
        else:
            lines.append("\n✗ Preprocessor compatibility issues found:")
            for issue in issues:
                lines.append(f"  - {issue}")

        # Feature spec
        lines.append(f"\nFeature Specification:")
        lines.append(f"  Audio: {self.spec.audio_dim}D")
        lines.append(f"  Video: {self.spec.video_dim}D")
        lines.append(f"  Text: {self.spec.text_dim}D")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


class FeatureCheckpoint:
    """Save and restore preprocessors for exact reproducibility."""

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_preprocessors(
        self,
        preprocessors: Dict[str, FeaturePreprocessor],
        experiment_name: str,
    ) -> None:
        """Save preprocessors for later use.

        Args:
            preprocessors: Dict of modality -> preprocessor
            experiment_name: Name for this experiment
        """
        exp_dir = self.checkpoint_dir / experiment_name / "preprocessors"
        exp_dir.mkdir(parents=True, exist_ok=True)

        for modality, prep in preprocessors.items():
            prep_path = exp_dir / f"{modality}_preprocessor.pkl"
            prep.save(prep_path)

    def load_preprocessors(
        self,
        experiment_name: str,
    ) -> Dict[str, FeaturePreprocessor]:
        """Load preprocessors from checkpoint.

        Args:
            experiment_name: Experiment name

        Returns:
            Dict of modality -> preprocessor
        """
        exp_dir = self.checkpoint_dir / experiment_name / "preprocessors"
        
        if not exp_dir.exists():
            raise FileNotFoundError(f"Preprocessor checkpoint not found: {exp_dir}")

        preprocessors = {}
        for modality in ["audio", "video", "text"]:
            prep_path = exp_dir / f"{modality}_preprocessor.pkl"
            if prep_path.exists():
                preprocessors[modality] = FeaturePreprocessor.load(prep_path)

        return preprocessors

    def save_feature_spec(
        self,
        spec: FeatureSpecification,
        experiment_name: str,
    ) -> None:
        """Save feature specification."""
        import json
        
        exp_dir = self.checkpoint_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        spec_path = exp_dir / "feature_spec.json"
        with open(spec_path, "w") as f:
            json.dump(spec.to_dict(), f, indent=2)

    def load_feature_spec(self, experiment_name: str) -> FeatureSpecification:
        """Load feature specification."""
        import json
        
        spec_path = self.checkpoint_dir / experiment_name / "feature_spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"Feature spec not found: {spec_path}")
        
        with open(spec_path, "r") as f:
            spec_dict = json.load(f)
        
        # Reconstruct FeatureSpecification from dict
        from src.features.specs import ModalitySpec
        
        audio_spec = ModalitySpec.from_dict(spec_dict["audio"])
        video_spec = ModalitySpec.from_dict(spec_dict["video"])
        text_spec = ModalitySpec.from_dict(spec_dict["text"])
        
        return FeatureSpecification(audio_spec, video_spec, text_spec)
