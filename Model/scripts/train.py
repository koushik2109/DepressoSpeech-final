import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset.builder import DatasetBuilder
from src.dataset.collate import multimodal_collate_fn
from src.models.multimodal_model import MultimodalDepressionModel
from src.training.trainer import Trainer
from src.utils.logging import configure_logging


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ModelV2 multimodal depression model")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = configure_logging(Path(config.get("logging", {}).get("log_dir", "logs")), experiment_name=config.get("logging", {}).get("experiment_name", "trimodal"))
    logger = logging.getLogger(__name__)

    train_dataset = DatasetBuilder.build_from_csv(
        split_csv_path=Path(config["dataset"]["train_split"]),
        feature_dir=Path(config["dataset"]["feature_dir"]),
        id_column=config["dataset"].get("id_column", "participant_id"),
        label_column=config["dataset"].get("label_column", "phq_total"),
        augment=True,
    )
    val_dataset = DatasetBuilder.build_from_csv(
        split_csv_path=Path(config["dataset"]["val_split"]),
        feature_dir=Path(config["dataset"]["feature_dir"]),
        id_column=config["dataset"].get("id_column", "participant_id"),
        label_column=config["dataset"].get("label_column", "phq_total"),
        augment=False,
    )

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

    model_config = config.get("model", {})
    model = MultimodalDepressionModel(**model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 3e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"].get("lr_factor", 0.5)),
        patience=int(config["training"].get("lr_patience", 3)),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config["training"],
    )
    logger.info("Starting ModelV2 training")
    history = trainer.train()
    logger.info(f"Training complete: {history}")


if __name__ == "__main__":
    main()
