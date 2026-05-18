import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, roc_auc_score


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    covariance = np.mean((y_true - true_mean) * (y_pred - pred_mean))
    true_var = np.var(y_true)
    pred_var = np.var(y_pred)
    return (2 * covariance) / (true_var + pred_var + (true_mean - pred_mean) ** 2 + epsilon)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "ccc": float(concordance_correlation_coefficient(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict[str, float]:
    binary = (y_true >= 10).astype(int)
    preds = (y_pred_probs >= 0.5).astype(int)
    auc = roc_auc_score(binary, y_pred_probs) if len(np.unique(binary)) > 1 else 0.5
    return {
        "f1": float(f1_score(binary, preds, zero_division=0)),
        "roc_auc": float(auc),
    }
