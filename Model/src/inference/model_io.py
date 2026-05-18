from pathlib import Path
from typing import Any, Dict
import torch


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: Path, device: str = "cpu") -> None:
    state = torch.load(path, map_location=device)
    # Support full training checkpoints that wrap the model state dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state
    # Load permissively to allow checkpoints saved from slightly different
    # training wrappers or model definitions. Report missing/unexpected keys.
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"load_checkpoint: missing keys: {result.missing_keys}")
        print(f"load_checkpoint: unexpected keys: {result.unexpected_keys}")


def safe_tensor(item: Any) -> Any:
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy()
    return item


def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float) and (value != value or value == float("inf") or value == float("-inf")):
            sanitized[key] = 0.0
        else:
            sanitized[key] = value
    return sanitized
