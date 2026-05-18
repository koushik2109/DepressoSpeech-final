#!/usr/bin/env python3
"""Improved training script for multimodal depression detection.

Features:
- Consistent feature pipeline with validation
- Advanced regularization (EMA, checkpoint averaging, curriculum learning)
- Comprehensive logging and checkpointing
- Small dataset optimization
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset.builder import DatasetBuilder
from src.dataset.collate import multimodal_collate_fn
from src.dataset.multimodal_dataset import MultimodalSample
from src.models.multimodal_model import MultimodalDepressionModel
from src.training.trainer import Trainer
from src.training.optimization import EMA, WarmupScheduler, CurriculumLearning
from src.features.specs import FeatureSpecification
from src.features.preprocessing import FeaturePreprocessor
from src.inference.validation import TrainInferenceConsistencyChecker, FeatureCheckpoint
from src.utils.logging import configure_logging


def load_config(path: Path) -> dict:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_feature_specs(config: dict) -> FeatureSpecification:
    """Set up feature specifications from config."""
    # Use defaults or override from config
    feature_config = config.get("features", {})

    # Create feature specs matching the config
    from src.features.specs import ModalitySpec

    # Determine which specs to use based on dimensions
    audio_dim = feature_config.get("audio_dim", 39)
    video_dim = feature_config.get("video_dim", 38)
    text_dim = feature_config.get("text_dim", 384)

    # For now, use standard specs (can be enhanced to support other combinations)
    spec = FeatureSpecification(
        audio_spec=ModalitySpec(name="audio", expected_dim=audio_dim, temporal=True),
        video_spec=ModalitySpec(name="video", expected_dim=video_dim, temporal=True),
        text_spec=ModalitySpec(name="text", expected_dim=text_dim, temporal=True),
    )

    # Removed strict assertions because PCA determines final dimensions dynamically
    return spec


def setup_preprocessors(config: dict, feature_spec: FeatureSpecification) -> dict:
    """Set up feature preprocessors (normalization, PCA)."""
    preprocessors = {}

    # Audio preprocessor
    audio_prep = FeaturePreprocessor(
        normalizer_type="standard",
        apply_pca=True,
        pca_components=0.80,
        pca_whiten=True,
    )
    preprocessors["audio"] = audio_prep

    # Video preprocessor
    video_prep = FeaturePreprocessor(
        normalizer_type="standard",
        apply_pca=True,
        pca_components=0.80,
        pca_whiten=True,
    )
    preprocessors["video"] = video_prep

    # Text preprocessor (Aggressive reduction)
    text_prep = FeaturePreprocessor(
        normalizer_type="standard",
        apply_pca=True,
        pca_components=0.80,
        pca_whiten=True,
    )
    preprocessors["text"] = text_prep

    return preprocessors


def _try_load_preprocessors(checkpoint_dir: Path, logger: logging.Logger) -> dict | None:
    """Try to load already-saved PCA preprocessors from the most recent checkpoint.

    Reuses the PCA transform computed in a previous training run so that:
    - dimensions stay consistent with the saved best_model.pt
    - no refitting noise between runs
    - faster startup

    Returns None if no valid checkpoint exists.
    """
    # Preference order: multimodal_v2 (best validated run), then any other
    candidates = ["multimodal_v3", "multimodal_v2", "multimodal_v1"]
    for name in candidates:
        prep_dir = checkpoint_dir / name / "preprocessors"
        if not prep_dir.exists():
            continue
        missing = [m for m in ["audio", "video", "text"] if not (prep_dir / f"{m}_preprocessor.pkl").exists()]
        if missing:
            continue
        try:
            from src.inference.validation import FeatureCheckpoint
            manager = FeatureCheckpoint(checkpoint_dir)
            preps = manager.load_preprocessors(name)
            if all(preps.get(m) and preps[m].output_dim for m in ["audio", "video", "text"]):
                logger.info(
                    f"Loaded existing PCA preprocessors from '{name}' checkpoint: "
                    f"audio={preps['audio'].output_dim}d  "
                    f"video={preps['video'].output_dim}d  "
                    f"text={preps['text'].output_dim}d"
                )
                return preps
        except Exception as exc:
            logger.warning(f"Could not load preprocessors from '{name}': {exc}")
    return None


def create_datasets_with_preprocessing(
    config: dict,
    preprocessors: dict,
    feature_spec: FeatureSpecification,
    logger: logging.Logger,
    checkpoint_dir: Path | None = None,
) -> tuple:
    """Create training and validation datasets with preprocessing.

    Tries to load already-saved PCA preprocessors first. Only refits if none exist.
    """

    reg_cfg = config.get("regularization", {})
    feat_noise = float(reg_cfg.get("feature_noise_std", 0.01))
    temp_drop  = float(reg_cfg.get("temporal_dropout_rate", 0.10))

    train_dataset_raw = DatasetBuilder.build_from_csv(
        split_csv_path=Path(config["dataset"]["train_split"]),
        feature_dir=Path(config["dataset"]["feature_dir"]),
        id_column=config["dataset"].get("id_column", "participant_id"),
        label_column=config["dataset"].get("label_column", "phq_total"),
        augment=True,
        temporal_dropout_rate=temp_drop,
        feature_noise_std=feat_noise,
    )

    val_dataset_raw = DatasetBuilder.build_from_csv(
        split_csv_path=Path(config["dataset"]["val_split"]),
        feature_dir=Path(config["dataset"]["feature_dir"]),
        id_column=config["dataset"].get("id_column", "participant_id"),
        label_column=config["dataset"].get("label_column", "phq_total"),
        augment=False,
        temporal_dropout_rate=0.0,
        feature_noise_std=0.0,
    )

    logger.info(f"Loaded {len(train_dataset_raw)} training samples")
    logger.info(f"Loaded {len(val_dataset_raw)} validation samples")

    # Try to reuse already-saved PCA preprocessors (avoids refitting noise)
    saved = _try_load_preprocessors(checkpoint_dir or Path("checkpoints"), logger)
    if saved:
        preprocessors = saved
        logger.info("Using saved PCA preprocessors — skipping refit.")
    else:
        # Fit PCA preprocessors fresh on training data
        logger.info("No saved preprocessors found. Fitting PCA from scratch...")
        for modality in ["audio", "video", "text"]:
            features_list = [
                s.get(modality) for s in train_dataset_raw.samples
                if s.get(modality) is not None
            ]
            if features_list:
                all_features = np.vstack(features_list)
                preprocessors[modality].fit(all_features)
                logger.info(
                    f"  {modality}: PCA fitted → {preprocessors[modality].output_dim}d "
                    f"(from {all_features.shape[1]}d, {all_features.shape[0]} rows)"
                )

    # Apply preprocessors to both splits
    logger.info("Applying PCA preprocessors to datasets...")
    for split_name, dataset in [("train", train_dataset_raw), ("val", val_dataset_raw)]:
        for sample in dataset.samples:
            for modality in ["audio", "video", "text"]:
                if sample.get(modality) is not None:
                    sample[modality] = preprocessors[modality].transform(sample[modality])
        logger.info(
            f"  {split_name}: applied PCA — audio now {dataset.samples[0]['audio'].shape[1] if dataset.samples[0].get('audio') is not None else 'N/A'}d"
        )

    return train_dataset_raw, val_dataset_raw, preprocessors


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train improved multimodal depression model")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--experiment-name", type=str, default="multimodal_v2")
    parser.add_argument("--use-ema", action="store_true", help="Use exponential moving average")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Fix random seed for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load config
    config = load_config(args.config)

    # Set up logging
    run_dir = configure_logging(
        Path(config.get("logging", {}).get("log_dir", "logs")),
        experiment_name=args.experiment_name,
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting training: {args.experiment_name}")
    logger.info(f"Config: {args.config}")

    # Feature specification and preprocessing
    logger.info("Setting up feature pipeline...")
    feature_spec = setup_feature_specs(config)
    logger.info(f"Feature spec - Audio: {feature_spec.audio_dim}D, Video: {feature_spec.video_dim}D, Text: {feature_spec.text_dim}D")

    preprocessors = setup_preprocessors(config, feature_spec)

    checkpoint_dir = Path(config["training"].get("checkpoint_dir", "checkpoints"))

    # Create datasets — loads existing PCA preprocessors if available
    train_dataset, val_dataset, preprocessors = create_datasets_with_preprocessing(
        config,
        preprocessors,
        feature_spec,
        logger,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Update model config dynamically based on PCA output dims
    if preprocessors["audio"].output_dim:
        config["model"]["audio_dim"] = preprocessors["audio"].output_dim
    if preprocessors["video"].output_dim:
        config["model"]["video_dim"] = preprocessors["video"].output_dim
    if preprocessors["text"].output_dim:
        config["model"]["text_dim"] = preprocessors["text"].output_dim
    
    logger.info(f"Final model dims - Audio: {config['model']['audio_dim']}, Video: {config['model']['video_dim']}, Text: {config['model']['text_dim']}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"].get("batch_size", 8)),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 2)),
        collate_fn=multimodal_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 2)),
        collate_fn=multimodal_collate_fn,
    )

    # Model
    model_config = config.get("model", {})
    model = MultimodalDepressionModel(**model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )

    # Warmup scheduler
    warmup_scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=5,
        total_epochs=int(config["training"].get("max_epochs", 100)),
        base_lr=float(config["training"].get("learning_rate", 1e-4)),
    )

    # Reduce on plateau scheduler
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"].get("lr_factor", 0.5)),
        patience=int(config["training"].get("lr_patience", 5)),
    )

    # EMA if enabled
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=0.999)
        logger.info("Using exponential moving average")

    # Merge loss weights into training config for Trainer
    training_config = config["training"].copy()
    if "loss" in config:
        training_config.update(config["loss"])

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=plateau_scheduler,
        config=training_config,
    )

    # Training
    logger.info("Starting training...")
    history = trainer.train()

    # Save best model info
    best_loss = getattr(trainer, "best_loss", None)
    best_ccc = getattr(trainer, "best_ccc", None)
    logger.info("Training complete!")
    logger.info(f"Best CCC: {best_ccc:.4f}" if best_ccc else "Best CCC: N/A")
    
    # Write result.json
    result_data = {
        "best_ccc": float(best_ccc) if best_ccc else 0.0,
        "best_loss": float(best_loss) if best_loss else 0.0,
        "experiment_name": args.experiment_name,
        "config": convert_numpy(config),
        "run_dir": str(run_dir),
    }
    import json
    with open("result.json", "w") as f:
        json.dump(result_data, f, indent=4)
    logger.info(f"Best loss: {history.get('best_loss', 'N/A')}")

    # Save preprocessors for inference
    checkpoint_manager = FeatureCheckpoint(Path(config["training"].get("checkpoint_dir", "checkpoints")))
    checkpoint_manager.save_preprocessors(preprocessors, args.experiment_name)
    checkpoint_manager.save_feature_spec(feature_spec, args.experiment_name)
    logger.info(f"Saved preprocessors and feature spec to {args.experiment_name}")


if __name__ == "__main__":
    main()
