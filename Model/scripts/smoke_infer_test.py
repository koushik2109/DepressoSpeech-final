#!/usr/bin/env python3
"""Smoke test: load inference config and instantiate ModelV2Inferencer using CPU."""
import sys
from pathlib import Path
import traceback

from src.utils.config import load_config


def main():
    try:
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / "configs" / "inference.yaml"
        cfg = load_config(cfg_path)
        # Force CPU to avoid CUDA/device autodetection issues
        cfg.setdefault("inference", {})["device"] = "cpu"

        checkpoint = Path(cfg["artifacts"]["checkpoint_path"]) if "artifacts" in cfg else Path("checkpoints/best_model.pt")
        if not checkpoint.is_absolute():
            checkpoint = (repo_root / checkpoint).resolve()

        print("Config loaded. Checkpoint:", checkpoint)
        print("Exists:", checkpoint.exists())

        from src.inference.inferencer import ModelV2Inferencer

        model_cfg = cfg.get("model", {})
        infer = ModelV2Inferencer(checkpoint, model_cfg, device="cpu")
        print("Model instantiated on device:", infer.device)
        print("Smoke test OK")
    except Exception:
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
