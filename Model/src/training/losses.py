import torch
import torch.nn as nn


class ConcordanceCorrelationCoefficientLoss(nn.Module):
    """Concordance Correlation Coefficient loss for regression.
    
    Better than MSE for capturing correlation consistency.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mean_pred = predictions.mean(dim=0)
        mean_true = targets.mean(dim=0)
        var_pred = predictions.var(dim=0, unbiased=False)
        var_true = targets.var(dim=0, unbiased=False)
        covariance = ((predictions - mean_pred) * (targets - mean_true)).mean(dim=0)
        ccc = (2.0 * covariance) / (var_pred + var_true + (mean_pred - mean_true).pow(2) + 1e-6)
        return 1.0 - ccc.mean()


class ConfidenceAwareLoss(nn.Module):
    """Confidence-aware regression loss.
    
    Uses per-sample confidence scores to weight regression targets.
    Encourages the model to be confident when predictions are accurate.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Compute confidence-aware loss.
        
        Args:
            predictions: Model predictions [batch_size]
            targets: Ground truth targets [batch_size]
            confidence: Confidence scores [batch_size]
            
        Returns:
            Weighted loss with confidence regularization
        """
        # Base MSE loss
        mse = (predictions - targets).pow(2)
        
        # Weight by confidence (downweight uncertain predictions)
        confident_loss = mse * confidence
        
        # Also penalize high confidence on bad predictions
        error_magnitude = (predictions - targets).abs()
        confidence_penalty = confidence * error_magnitude
        
        return (confident_loss + 0.1 * confidence_penalty).mean()


class EntropyRegularization(nn.Module):
    """Entropy regularization to encourage multimodal diversity.

    Target is log(3) ≈ 1.099 — the maximum entropy for 3 modalities, meaning
    truly balanced (uniform) gate weights.  Penalises both collapse to one
    modality and over-uniform (no learning).
    """
    def __init__(self, target_entropy: float = 1.05):
        super().__init__()
        self.target_entropy = target_entropy

    def forward(self, entropy: torch.Tensor) -> torch.Tensor:
        entropy_loss = (entropy - self.target_entropy).pow(2)
        return entropy_loss


class GateBalanceLoss(nn.Module):
    """Penalises deviation of gate weights from the uniform distribution (1/N).

    For 3 modalities the ideal unbiased weights are [1/3, 1/3, 1/3].
    This loss prevents any one modality from monopolising the fusion gate
    which would make the model fragile during inference when modalities are
    missing.

    Loss = mean over batch of ||w - 1/N||²
    """

    def __init__(self, num_modalities: int = 3):
        super().__init__()
        self.num_modalities = num_modalities
        self.uniform = 1.0 / num_modalities

    def forward(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate_weights: [batch_size, num_modalities] softmax gate weights

        Returns:
            Scalar balance loss
        """
        deviation = (gate_weights - self.uniform).pow(2)
        return deviation.mean()


class TextGatePreferenceLoss(nn.Module):
    """Penalises when the text gate weight falls below a target floor.

    Encourages the model to rely more on linguistic features (text) which are
    the most predictive modality for PHQ-8 depression detection.
    Loss = mean(relu(target - w_text))^2  — only penalises *below* target.
    """

    def __init__(self, target: float = 0.40):
        super().__init__()
        self.target = target

    def forward(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate_weights: [batch_size, 3] — order: audio(0), video(1), text(2)
        Returns:
            Scalar loss (0 when text_gate >= target)
        """
        text_gate = gate_weights[:, 2]  # text is index 2
        shortfall = torch.relu(self.target - text_gate)
        return shortfall.pow(2).mean()


class MultitaskLoss(nn.Module):
    """Multitask loss combining regression, classification, question-level,
    entropy regularisation, gate balance and confidence objectives."""

    def __init__(
        self,
        regression_weight: float = 1.0,
        classification_weight: float = 0.5,
        question_weight: float = 1.0,
        entropy_weight: float = 0.01,
        ccc_weight: float = 0.2,
        confidence_weight: float = 0.1,
        gate_balance_weight: float = 0.5,
        text_gate_weight: float = 0.0,
        text_gate_target: float = 0.40,
    ):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.question_weight = question_weight
        self.entropy_weight = entropy_weight
        self.ccc_weight = ccc_weight
        self.confidence_weight = confidence_weight
        self.gate_balance_weight = gate_balance_weight
        self.text_gate_weight = text_gate_weight

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ccc = ConcordanceCorrelationCoefficientLoss()
        self.confidence_loss = ConfidenceAwareLoss()
        self.entropy_reg = EntropyRegularization(target_entropy=1.05)
        self.gate_balance = GateBalanceLoss(num_modalities=3)
        self.text_gate_pref = TextGatePreferenceLoss(target=text_gate_target)

    def forward(
        self,
        phq_total_pred: torch.Tensor,
        phq_total_target: torch.Tensor,
        phq_questions_pred: torch.Tensor,
        phq_questions_target: torch.Tensor,
        classification_pred: torch.Tensor,
        classification_target: torch.Tensor,
        entropy_loss: torch.Tensor,
        confidence_pred: torch.Tensor | None = None,
        gate_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total_loss = 0.0

        # CCC is the primary metric — give it the highest weight
        if self.ccc_weight > 0.0:
            total_loss += self.ccc_weight * self.ccc(phq_total_pred, phq_total_target)

        # Auxiliary regression (MSE) for stable gradient scale
        regression_loss = self.mse(phq_total_pred, phq_total_target)
        total_loss += self.regression_weight * regression_loss

        # Per-question regression
        total_loss += self.question_weight * self.mse(phq_questions_pred, phq_questions_target)

        # Binary classification (depressed / not depressed)
        total_loss += self.classification_weight * self.bce(classification_pred, classification_target)

        # Entropy regularisation — steer gates towards balanced attention
        total_loss += self.entropy_weight * self.entropy_reg(entropy_loss)

        # Gate balance — penalise any gate collapsing to near-zero
        if gate_weights is not None and self.gate_balance_weight > 0.0:
            total_loss += self.gate_balance_weight * self.gate_balance(gate_weights)

        # Text gate preference — encourage text >= target floor
        if gate_weights is not None and self.text_gate_weight > 0.0:
            total_loss += self.text_gate_weight * self.text_gate_pref(gate_weights)

        # Confidence-aware loss
        if confidence_pred is not None and self.confidence_weight > 0.0:
            if confidence_pred.dim() > 1:
                confidence_score = confidence_pred.mean(dim=-1)
            else:
                confidence_score = confidence_pred
            confidence_loss = self.confidence_loss(phq_total_pred, phq_total_target, confidence_score)
            total_loss += self.confidence_weight * confidence_loss

        return total_loss

