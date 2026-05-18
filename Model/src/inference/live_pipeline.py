"""Live inference pipeline for real-time multimodal behavioral prediction.

Handles end-to-end inference from raw inputs to model predictions with
full consistency with training pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.inference.model_io import load_checkpoint, sanitize_payload
from src.inference.runtime_extractors import (
    LiveInferenceFeatureExtractor,
    OpenFaceRuntimeExtractor,
    AudioFeatureRuntimeExtractor,
    TextFeatureRuntimeExtractor,
)
from src.inference.validation import (
    FeatureValidator,
    TrainInferenceConsistencyChecker,
)
from src.models.multimodal_model import MultimodalDepressionModel
from src.features.specs import FeatureSpecification
from src.features.preprocessing import FeaturePreprocessor


class LiveInferencePipeline:
    """End-to-end live inference from raw inputs.

    Handles:
    - Feature extraction from raw video/audio/text
    - Preprocessing and normalization
    - Feature validation
    - Model inference
    - Result post-processing
    """

    def __init__(
        self,
        checkpoint_path: Path,
        model_config: Dict[str, Any],
        feature_spec: FeatureSpecification,
        preprocessors: Optional[Dict[str, FeaturePreprocessor]] = None,
        device: str = "auto",
    ):
        """Initialize live inference pipeline.

        Args:
            checkpoint_path: Path to trained model checkpoint
            model_config: Model configuration dict
            feature_spec: Feature specification for validation
            preprocessors: Optional preprocessors (normalization, PCA)
            device: Device for inference ("auto", "cuda", "cpu")
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.feature_spec = feature_spec
        self.preprocessors = preprocessors or {}

        # Load model
        self.model = MultimodalDepressionModel(**model_config)
        load_checkpoint(self.model, checkpoint_path, device=str(self.device))
        self.model.to(self.device).eval()

        # Feature extractors
        resnet_pca = self.preprocessors.get("video_resnet_pca") if "video" in self.preprocessors else None
        audio_normalizer = (
            self.preprocessors["audio"].normalizer
            if "audio" in self.preprocessors else None
        )

        self.feature_extractor = LiveInferenceFeatureExtractor(
            openface_pca=None,  # OpenFace is not PCA-reduced
            resnet_pca=resnet_pca,
            audio_normalizer=audio_normalizer,
            device=str(self.device),
        )

        # Validator
        self.validator = FeatureValidator(feature_spec)

    def extract_video_features(
        self,
        video_path: Path | str,
        openface_csv_path: Optional[Path | str] = None,
    ) -> np.ndarray:
        """Extract video features from OpenFace CSV.

        Args:
            video_path: Path to video file (not used if CSV provided)
            openface_csv_path: Path to pre-computed OpenFace CSV

        Returns:
            Feature array [num_frames, feature_dim]
        """
        if openface_csv_path:
            return self.feature_extractor.openface_extractor.extract(openface_csv_path)
        else:
            raise ValueError(
                "OpenFace extraction from raw video requires OpenFace binary. "
                "Please provide pre-computed OpenFace CSV."
            )

    def extract_audio_features(self, audio_path: Path | str) -> np.ndarray:
        """Extract audio features from file.

        Args:
            audio_path: Path to audio file

        Returns:
            Feature array [num_frames, feature_dim]
        """
        return self.feature_extractor.extract_audio(audio_path)["mfcc"]

    def extract_text_features(self, text: str) -> np.ndarray:
        """Extract text features from transcript.

        Args:
            text: Transcript text

        Returns:
            Feature array [num_chunks, feature_dim]
        """
        return self.feature_extractor.extract_text(text)

    def _prepare_feature_tensor(
        self,
        features: Optional[np.ndarray],
        expected_dim: int,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Prepare feature tensor for model inference.

        Args:
            features: Feature array or None
            expected_dim: Expected feature dimension

        Returns:
            Tuple of (feature_tensor, mask)
        """
        if features is None:
            return None, torch.zeros(1, 0, dtype=torch.bool, device=self.device)

        # Validate dimensions
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got {features.ndim}D")
        if features.shape[1] != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch. "
                f"Expected {expected_dim}, got {features.shape[1]}"
            )

        # Convert to tensor
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)  # [1, time, dim]
        mask = torch.ones(1, features.shape[0], dtype=torch.bool, device=self.device)

        return tensor, mask

    def predict_single(
        self,
        audio_path: Optional[Path | str] = None,
        video_openface_csv: Optional[Path | str] = None,
        transcript: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run inference on a single sample.

        Args:
            audio_path: Path to audio file
            video_openface_csv: Path to OpenFace CSV (video features)
            transcript: Transcript text
            verbose: Whether to print detailed output

        Returns:
            Dictionary with predictions and metadata
        """
        # Extract features
        features = {}
        try:
            if audio_path:
                features["audio"] = self.extract_audio_features(audio_path)
                if verbose:
                    print(f"  Audio features: {features['audio'].shape}")
            
            if video_openface_csv:
                features["video"] = self.extract_video_features(
                    None,
                    video_openface_csv,
                )
                if verbose:
                    print(f"  Video features: {features['video'].shape}")
            
            if transcript:
                features["text"] = self.extract_text_features(transcript)
                if verbose:
                    print(f"  Text features: {features['text'].shape}")
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {e}")

        # Validate features
        validation = self.validator.validate_batch(features)
        if not validation.is_valid:
            print(validation.report())
            raise ValueError("Feature validation failed")

        # Prepare tensors
        audio_tensor, audio_mask = self._prepare_feature_tensor(
            features.get("audio"),
            self.feature_spec.audio_dim,
        )
        video_tensor, video_mask = self._prepare_feature_tensor(
            features.get("video"),
            self.feature_spec.video_dim,
        )
        text_tensor, text_mask = self._prepare_feature_tensor(
            features.get("text"),
            self.feature_spec.text_dim,
        )

        # Run inference
        with torch.no_grad():
            output = self.model(
                audio_tensor,
                video_tensor,
                text_tensor,
                audio_mask,
                video_mask,
                text_mask,
            )

        # Post-process outputs
        phq_total = float(output["phq_total"].cpu().numpy()[0])
        phq_questions = output["phq_questions"].cpu().numpy()[0].tolist()
        classification = float(torch.sigmoid(output["classification"]).cpu().numpy()[0])
        confidence = float(output["confidence"].cpu().numpy()[0])

        # Modality scores
        modality_scores = {}
        if "modality_scores" in output:
            for key, val in output["modality_scores"].items():
                if isinstance(val, torch.Tensor):
                    modality_scores[key] = float(val.cpu().numpy())
                else:
                    modality_scores[key] = float(val)

        # Modality confidence
        modality_confidence = {}
        if "modality_confidence" in output:
            conf_array = output["modality_confidence"].cpu().numpy()[0]
            modality_confidence = {
                "audio": float(conf_array[0]),
                "video": float(conf_array[1]),
                "text": float(conf_array[2]),
            }

        entropy = float(output["entropy"].cpu().numpy())

        # Binary classification (PHQ >= 10)
        depression_binary = phq_total >= 10

        return {
            "phq_total": phq_total,
            "phq_questions": phq_questions,
            "classification_prob": classification,
            "depression_likely": depression_binary,
            "confidence": confidence,
            "modality_scores": modality_scores,
            "modality_confidence": modality_confidence,
            "entropy": entropy,
            "input_features": {
                "audio_frames": features.get("audio").shape[0] if "audio" in features else 0,
                "video_frames": features.get("video").shape[0] if "video" in features else 0,
                "text_chunks": features.get("text").shape[0] if "text" in features else 0,
            },
        }

    def predict_batch(
        self,
        samples: list[Dict[str, Any]],
        verbose: bool = False,
    ) -> list[Dict[str, Any]]:
        """Run inference on multiple samples.

        Args:
            samples: List of dicts with "audio_path", "video_openface_csv", "transcript"
            verbose: Whether to print progress

        Returns:
            List of prediction dicts
        """
        results = []
        for i, sample in enumerate(samples):
            if verbose:
                print(f"Inference {i+1}/{len(samples)}...")
            
            result = self.predict_single(
                audio_path=sample.get("audio_path"),
                video_openface_csv=sample.get("video_openface_csv"),
                transcript=sample.get("transcript"),
                verbose=verbose,
            )
            results.append(result)

        return results

    def predict_and_explain(
        self,
        audio_path: Optional[Path | str] = None,
        video_openface_csv: Optional[Path | str] = None,
        transcript: Optional[str] = None,
    ) -> str:
        """Run inference and generate human-readable report.

        Args:
            audio_path: Path to audio file
            video_openface_csv: Path to OpenFace CSV
            transcript: Transcript text

        Returns:
            Formatted report string
        """
        result = self.predict_single(
            audio_path=audio_path,
            video_openface_csv=video_openface_csv,
            transcript=transcript,
            verbose=False,
        )

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("MULTIMODAL BEHAVIORAL ASSESSMENT")
        lines.append("=" * 70)

        # PHQ Score
        lines.append(f"\nPHQ-9 Total Score: {result['phq_total']:.1f}/27")
        if result["depression_likely"]:
            lines.append("  → MODERATE TO SEVERE DEPRESSION (PHQ-9 ≥ 10)")
        else:
            lines.append("  → MINIMAL TO MILD DEPRESSION (PHQ-9 < 10)")

        # Individual questions
        if result["phq_questions"]:
            lines.append("\nPer-Question Scores:")
            questions = [
                "Little interest/pleasure",
                "Feeling down",
                "Sleep problems",
                "Fatigue/low energy",
                "Appetite changes",
                "Feeling bad about self",
                "Concentration problems",
                "Psychomotor changes",
            ]
            for i, (q, score) in enumerate(zip(questions, result["phq_questions"])):
                lines.append(f"  Q{i+1}: {q:30s} {score:.1f}")

        # Classification confidence
        lines.append(f"\nClassification Confidence: {result['classification_prob']:.2%}")
        lines.append(f"Prediction Confidence: {result['confidence']:.2%}")

        # Modality contributions
        if result["modality_scores"]:
            lines.append("\nModality Contributions to Prediction:")
            for modality in ["audio", "video", "text"]:
                if modality in result["modality_scores"]:
                    score = result["modality_scores"][modality]
                    conf = result["modality_confidence"].get(modality, 0.0)
                    lines.append(f"  {modality.upper():8s}: weight={score:.2%}, confidence={conf:.2%}")

        lines.append(f"\nFusion Entropy: {result['entropy']:.4f}")
        lines.append(f"Input Features:")
        for feature_type, count in result["input_features"].items():
            lines.append(f"  {feature_type}: {count}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
