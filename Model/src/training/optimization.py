"""Training utilities for small dataset optimization.

Implements advanced techniques:
- Exponential Moving Average (EMA) weights
- Checkpoint averaging
- Curriculum learning
- Gradient accumulation
- Warmup scheduling
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List
import copy


class EMA(nn.Module):
    """Exponential Moving Average for model weights.

    Maintains a running average of model weights for better generalization,
    especially important for small datasets.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Initialize EMA wrapper.

        Args:
            model: Model to apply EMA to
            decay: EMA decay rate (higher = slower updates)
        """
        super().__init__()
        self.decay = decay
        self.model = model
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

        for param in self.shadow.parameters():
            param.requires_grad = False

    def update(self) -> None:
        """Update shadow weights with EMA."""
        for model_param, shadow_param in zip(self.model.parameters(), self.shadow.parameters()):
            shadow_param.data = self.decay * shadow_param.data + (1 - self.decay) * model_param.data.detach()

    def forward(self, *args, **kwargs):
        """Forward pass through shadow model (during inference)."""
        return self.shadow(*args, **kwargs)

    def swap(self) -> None:
        """Swap model and shadow parameters for evaluation."""
        self.model, self.shadow = self.shadow, self.model

    def restore(self) -> None:
        """Restore original model and shadow after evaluation."""
        self.swap()


class CheckpointAveraging:
    """Average weights from multiple checkpoints for improved generalization."""

    def __init__(self, num_checkpoints: int = 5):
        """Initialize checkpoint averaging.

        Args:
            num_checkpoints: Number of last checkpoints to average
        """
        self.num_checkpoints = num_checkpoints
        self.checkpoints: List[dict] = []

    def add_checkpoint(self, state_dict: dict) -> None:
        """Add checkpoint state dict.

        Args:
            state_dict: Model state dict to add
        """
        import copy
        self.checkpoints.append(copy.deepcopy(state_dict))
        # Keep only recent checkpoints
        if len(self.checkpoints) > self.num_checkpoints:
            self.checkpoints.pop(0)

    def get_averaged_weights(self) -> dict:
        """Compute average of checkpoint weights.

        Returns:
            Averaged state dict
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints to average")

        averaged = {}
        num_checkpoints = len(self.checkpoints)

        # Get keys from first checkpoint
        for key in self.checkpoints[0].keys():
            # Average weights
            weight_sum = torch.zeros_like(self.checkpoints[0][key])
            for ckpt in self.checkpoints:
                weight_sum += ckpt[key]
            averaged[key] = weight_sum / num_checkpoints

        return averaged


class WarmupScheduler:
    """Learning rate warmup + decay scheduler.

    Gradually increases LR during warmup phase, then decays.
    Important for small datasets to prevent divergence.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        base_lr: float = 1e-4,
        min_lr: float = 1e-6,
    ):
        """Initialize scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for warmup
            total_epochs: Total number of training epochs
            base_lr: Base learning rate
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.epoch = 0

    def step(self) -> None:
        """Update learning rate for current epoch."""
        if self.epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing decay
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.epoch += 1

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class CurriculumLearning:
    """Curriculum learning scheduler for small datasets.

    Gradually increases task difficulty (e.g., more modality dropout).
    Helps prevent overfitting on small datasets.
    """

    def __init__(
        self,
        initial_dropout_rate: float = 0.0,
        final_dropout_rate: float = 0.2,
        total_epochs: int = 50,
    ):
        """Initialize curriculum.

        Args:
            initial_dropout_rate: Starting dropout rate (easy)
            final_dropout_rate: Final dropout rate (hard)
            total_epochs: Total training epochs
        """
        self.initial_dropout = initial_dropout_rate
        self.final_dropout = final_dropout_rate
        self.total_epochs = total_epochs
        self.epoch = 0

    def get_current_difficulty(self) -> float:
        """Get current difficulty level (0 to 1)."""
        return self.epoch / self.total_epochs

    def get_modality_dropout_rate(self) -> float:
        """Get modality dropout rate for current epoch."""
        difficulty = self.get_current_difficulty()
        return self.initial_dropout + (self.final_dropout - self.initial_dropout) * difficulty

    def get_temporal_dropout_rate(self) -> float:
        """Get temporal dropout rate for current epoch."""
        difficulty = self.get_current_difficulty()
        # Start with less temporal dropout, increase over time
        return difficulty * 0.15

    def step(self) -> None:
        """Advance to next epoch."""
        self.epoch += 1


class GradientAccumulation:
    """Gradient accumulation for large effective batch sizes on small datasets."""

    def __init__(self, accumulation_steps: int = 4):
        """Initialize gradient accumulation.

        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_accumulate(self) -> bool:
        """Check if should accumulate gradient."""
        return (self.step_count + 1) % self.accumulation_steps != 0

    def should_step(self) -> bool:
        """Check if should perform optimizer step."""
        return (self.step_count + 1) % self.accumulation_steps == 0

    def step(self) -> None:
        """Advance accumulation counter."""
        self.step_count += 1

    def reset(self) -> None:
        """Reset accumulation counter."""
        self.step_count = 0


class MixupAugmentation:
    """Mixup data augmentation for small datasets.

    Creates interpolated samples between training examples.
    """

    def __init__(self, alpha: float = 0.2):
        """Initialize mixup.

        Args:
            alpha: Beta distribution parameter for mixing
        """
        self.alpha = alpha

    def __call__(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        target1: float,
        target2: float,
    ) -> tuple[torch.Tensor, float]:
        """Apply mixup to two samples.

        Args:
            features1: First feature tensor
            features2: Second feature tensor
            target1: First target value
            target2: Second target value

        Returns:
            Tuple of (mixed_features, mixed_target)
        """
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        mixed_features = lam * features1 + (1 - lam) * features2
        mixed_target = lam * target1 + (1 - lam) * target2
        return mixed_features, mixed_target


class ManifoldMixup:
    """Manifold Mixup: apply mixup in hidden layer space.

    More effective than input-space mixup for small datasets.
    """

    def __init__(self, alpha: float = 0.2):
        """Initialize manifold mixup.

        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(
        self,
        hidden1: torch.Tensor,
        hidden2: torch.Tensor,
        target1: float,
        target2: float,
    ) -> tuple[torch.Tensor, float]:
        """Apply mixup to hidden representations.

        Args:
            hidden1: First hidden representation
            hidden2: Second hidden representation
            target1: First target value
            target2: Second target value

        Returns:
            Tuple of (mixed_hidden, mixed_target)
        """
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        mixed_hidden = lam * hidden1 + (1 - lam) * hidden2
        mixed_target = lam * target1 + (1 - lam) * target2
        return mixed_hidden, mixed_target


class StochasticDepth(nn.Module):
    """Stochastic depth (drop path) for regularization.

    Randomly drops entire residual connections during training.
    """

    def __init__(self, drop_prob: float = 0.1):
        """Initialize stochastic depth.

        Args:
            drop_prob: Probability of dropping path
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth.

        Args:
            x: Main path output
            residual: Residual connection

        Returns:
            Output with stochastic depth applied
        """
        if not self.training or self.drop_prob == 0:
            return x + residual

        # Randomly drop path
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device)
        random_tensor = (random_tensor < keep_prob).float() / keep_prob
        return (x + residual) * random_tensor


class FocalLoss(nn.Module):
    """Focal loss for class imbalance in binary classification.

    Downweights easy examples, focuses on hard ones.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize focal loss.

        Args:
            alpha: Weighting factor in [0, 1]
            gamma: Focusing parameter >= 0
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            predictions: Logits [batch_size]
            targets: Binary targets [batch_size]

        Returns:
            Focal loss
        """
        # Convert logits to probabilities
        p = torch.sigmoid(predictions)

        # Compute cross entropy
        ce = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction="none")

        # Compute focal loss
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * ce

        return focal_loss.mean()
