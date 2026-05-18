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
from src.evaluation.evaluator import evaluate_predictions


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ModelV2 checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(Path(config.get("logging", {}).get("log_dir", "logs")))
    logger = logging.getLogger(__name__)

    val_dataset = DatasetBuilder.build_from_csv(
        split_csv_path=Path(config["dataset"]["val_split"]),
        feature_dir=Path(config["dataset"]["feature_dir"]),
        id_column=config["dataset"].get("id_column", "participant_id"),
        label_column=config["dataset"].get("label_column", "phq_total"),
        augment=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.get("inference", {}).get("batch_size", 4), shuffle=False, collate_fn=multimodal_collate_fn)

    model = MultimodalDepressionModel(**config["model"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    y_true, y_pred, q_true, q_pred, cls_true, cls_pred = [], [], [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            output = model(
                batch["audio"], batch["video"], batch["text"], batch["audio_mask"], batch["video_mask"], batch["text_mask"],
            )
            y_true.extend(batch["phq_total"].tolist())
            y_pred.extend(output["phq_total"].tolist())
            q_true.extend(batch["phq_questions"].tolist())
            q_pred.extend(output["phq_questions"].tolist())
            cls_true.extend(batch["classification"].tolist())
            cls_pred.extend(torch.sigmoid(output["classification"]).tolist())

    results = evaluate_predictions(y_true, y_pred, q_true, q_pred, cls_true, cls_pred)
    logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()
