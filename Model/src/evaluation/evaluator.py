import numpy as np
from typing import Sequence

from src.training.metrics import regression_metrics, classification_metrics


def evaluate_predictions(
    phq_total_true: Sequence[float],
    phq_total_pred: Sequence[float],
    phq_question_true: Sequence[Sequence[float]],
    phq_question_pred: Sequence[Sequence[float]],
    classification_true: Sequence[float],
    classification_pred_probs: Sequence[float],
) -> dict[str, float]:
    total_results = regression_metrics(np.array(phq_total_true), np.array(phq_total_pred))
    question_results = regression_metrics(
        np.array(phq_question_true).reshape(-1),
        np.array(phq_question_pred).reshape(-1),
    )
    classification_results = classification_metrics(
        np.array(classification_true), np.array(classification_pred_probs)
    )
    return {
        **{f"total_{k}": v for k, v in total_results.items()},
        **{f"question_{k}": v for k, v in question_results.items()},
        **classification_results,
    }
