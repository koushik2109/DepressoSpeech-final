#!/usr/bin/env python3
"""Example live inference script for multimodal depression detection.

Demonstrates how to use the LiveInferencePipeline for real-time predictions
from webcam/video/audio/transcript inputs.
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from src.inference.live_pipeline import LiveInferencePipeline
from src.features.specs import FeatureSpecification
from src.inference.validation import FeatureCheckpoint


def main() -> None:
    """Example usage of live inference pipeline."""
    parser = argparse.ArgumentParser(description="Live inference example")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name for loading preprocessors")
    parser.add_argument("--audio", type=Path, help="Path to audio file")
    parser.add_argument("--openface-csv", type=Path, help="Path to OpenFace CSV (video features)")
    parser.add_argument("--transcript", type=str, help="Transcript text")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Load feature spec and preprocessors
    checkpoint_manager = FeatureCheckpoint(Path("checkpoints"))
    
    try:
        feature_spec = checkpoint_manager.load_feature_spec(args.experiment)
        preprocessors = checkpoint_manager.load_preprocessors(args.experiment)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure to run training script with --experiment-name {args.experiment}")
        sys.exit(1)

    # Model config (same as training)
    model_config = {
        "audio_dim": feature_spec.audio_dim,
        "video_dim": feature_spec.video_dim,
        "text_dim": feature_spec.text_dim,
        "fusion_dim": 256,
        "num_questions": 8,
        "encoder_type": "transformer",
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.2,
        "fusion_mode": "hybrid",
    }

    # Initialize pipeline
    pipeline = LiveInferencePipeline(
        checkpoint_path=args.checkpoint,
        model_config=model_config,
        feature_spec=feature_spec,
        preprocessors=preprocessors,
        device=args.device,
    )

    if args.verbose:
        print(f"Model loaded from: {args.checkpoint}")
        print(f"Feature spec: audio={feature_spec.audio_dim}D, video={feature_spec.video_dim}D, text={feature_spec.text_dim}D")

    # Run inference
    try:
        result = pipeline.predict_single(
            audio_path=args.audio,
            video_openface_csv=args.openface_csv,
            transcript=args.transcript,
            verbose=args.verbose,
        )

        # Print results
        print(pipeline.predict_and_explain(
            audio_path=args.audio,
            video_openface_csv=args.openface_csv,
            transcript=args.transcript,
        ))

    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
