from __future__ import annotations

import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from torch.utils.data import DataLoader

from src.training.losses import MultitaskLoss
from src.training.metrics import regression_metrics
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        if value + self.min_delta < self.best:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: dict,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_epochs = config.get("max_epochs", 50)
        self.gradient_clip_norm = config.get("grad_clip_norm", 1.0)
        self.use_amp = config.get("use_amp", True)
        self.early_stopping = EarlyStopping(config.get("early_stopping_patience", 8), config.get("early_stopping_min_delta", 0.001))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.device.type == "cuda")
        self.loss_fn = MultitaskLoss(
            regression_weight=config.get("regression_weight", 1.0),
            classification_weight=config.get("classification_weight", 0.5),
            question_weight=config.get("question_weight", 1.0),
            entropy_weight=config.get("entropy_weight", 0.02),
            ccc_weight=config.get("ccc_weight", 0.2),
            confidence_weight=config.get("confidence_weight", 0.05),
            gate_balance_weight=config.get("gate_balance_weight", 0.5),
            text_gate_weight=config.get("text_gate_weight", 0.0),
            text_gate_target=config.get("text_gate_target", 0.40),
        )
        self.mixup_alpha = float(config.get("mixup_alpha", 0.0))
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # EMA for better generalization on small datasets
        from src.training.optimization import EMA
        self.ema = EMA(self.model, decay=0.999)
        self.best_ccc = -float("inf")

    def train(self) -> Dict[str, float]:
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_ccc": [],
            "val_rmse": [],
            "val_mae": [],
        }
        
        logger.info("Starting training loop...")
        best_path = self.checkpoint_dir / "best_model.pt"
        best_ccc = -float("inf")

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            
            # Use EMA weights for validation
            self.ema.swap()
            val_loss, val_metrics = self._validate_epoch()
            self.ema.restore()
            
            val_ccc = val_metrics.get("ccc", val_metrics.get("total_ccc", 0.0))
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_rmse"].append(val_metrics.get("rmse", 0.0))
            history["val_mae"].append(val_metrics.get("mae", 0.0))
            history["val_ccc"].append(val_ccc)

            gate_str = ""
            if "gate_audio" in val_metrics:
                gate_str = f" | Gates: A:{val_metrics['gate_audio']:.2f} V:{val_metrics['gate_video']:.2f} T:{val_metrics['gate_text']:.2f}"
            logger.info(
                f"Epoch {epoch:03d}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val CCC: {val_ccc:.4f} | "
                f"Val RMSE: {val_metrics.get('rmse', val_metrics.get('total_rmse', 0.0)):.4f}{gate_str}"
            )

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                
            if val_ccc > best_ccc:
                best_ccc = val_ccc
                self.best_ccc = best_ccc
                logger.info(f"  --> New best model saved with CCC: {val_ccc:.4f}")
                # Save EMA weights as the best model
                self.ema.swap()
                torch.save(self.model.state_dict(), best_path)
                self.ema.restore()
                
            if self.early_stopping.step(-val_ccc): # Early stopping based on CCC (negative because it expects min_delta to be lower)
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        return {"best_ccc": best_ccc, **history}

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = self._compute_batch_loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update EMA weights
            self.ema.update()
            
            total_loss += float(loss.detach().cpu())
        return total_loss / max(1, len(self.train_loader))

    def _validate_epoch(self) -> tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        y_true = []
        y_pred = []
        all_modality_scores = []
        with torch.no_grad():
            for batch in self.val_loader:
                output = self._forward(batch)
                ms_val = output.get("modality_scores")
                gate_w_val = None
                if ms_val is not None and isinstance(ms_val, dict):
                    raw_v = ms_val.get("raw_weights")
                    gate_w_val = raw_v if raw_v is not None else torch.stack(
                        [ms_val["audio"], ms_val["video"], ms_val["text"]], dim=1
                    ).to(self.device)
                loss = self.loss_fn(
                    output["phq_total"],
                    batch["phq_total"].to(self.device),
                    output["phq_questions"],
                    batch["phq_questions"].to(self.device),
                    output["classification"],
                    batch["classification"].to(self.device),
                    output["entropy"],
                    confidence_pred=output.get("modality_confidence"),
                    gate_weights=gate_w_val,
                )
                total_loss += float(loss.cpu())
                y_true.extend(batch["phq_total"].tolist())
                y_pred.extend(output["phq_total"].cpu().tolist())
                if "modality_scores" in output and output["modality_scores"] is not None:
                    ms = output["modality_scores"]
                    if isinstance(ms, dict):
                        gate_tensor = torch.stack([ms["audio"], ms["video"], ms["text"]], dim=1).cpu()
                        all_modality_scores.append(gate_tensor)
                    else:
                        all_modality_scores.append(ms.cpu())
        metrics = regression_metrics(y_true, y_pred)
        if all_modality_scores:
            avg_scores = torch.cat(all_modality_scores, dim=0).mean(dim=0).tolist()
            if len(avg_scores) >= 3:
                metrics["gate_audio"] = avg_scores[0]
                metrics["gate_video"] = avg_scores[1]
                metrics["gate_text"] = avg_scores[2]
        return total_loss / max(1, len(self.val_loader)), metrics

    def _compute_batch_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Optional mixup augmentation — interpolate between random sample pairs
        if self.model.training and self.mixup_alpha > 0.0 and len(batch["phq_total"]) > 1:
            batch = self._apply_mixup(batch)

        output = self._forward(batch)

        # Use raw_weights (has gradients) so gate_balance loss can backprop
        gate_weights = None
        ms = output.get("modality_scores")
        if ms is not None and isinstance(ms, dict):
            raw = ms.get("raw_weights")
            if raw is not None:
                gate_weights = raw
            else:
                gate_weights = torch.stack(
                    [ms["audio"], ms["video"], ms["text"]], dim=1
                ).to(self.device)

        loss = self.loss_fn(
            output["phq_total"],
            batch["phq_total"].to(self.device),
            output["phq_questions"],
            batch["phq_questions"].to(self.device),
            output["classification"],
            batch["classification"].to(self.device),
            output["entropy"],
            confidence_pred=output.get("modality_confidence"),
            gate_weights=gate_weights,
        )
        return loss

    def _apply_mixup(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Mix pairs of samples in the batch (label-preserving interpolation)."""
        import numpy as np
        B = batch["phq_total"].shape[0]
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        idx = torch.randperm(B)
        mixed = dict(batch)  # shallow copy
        for key in ("phq_total", "phq_questions", "classification"):
            mixed[key] = lam * batch[key] + (1 - lam) * batch[key][idx]
        # Feature tensors: mix along time dim with the same lambda
        for key in ("audio", "video", "text"):
            if batch.get(key) is not None:
                mixed[key] = lam * batch[key] + (1 - lam) * batch[key][idx]
        return mixed

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio = batch["audio"].to(self.device) if batch.get("audio") is not None else None
        video = batch["video"].to(self.device) if batch.get("video") is not None else None
        text = batch["text"].to(self.device) if batch.get("text") is not None else None
        audio_mask = batch["audio_mask"].to(self.device)
        video_mask = batch["video_mask"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)
        return self.model(audio, video, text, audio_mask, video_mask, text_mask)
