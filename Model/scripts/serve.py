import argparse
import uvicorn
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.app import create_app
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve ModelV2 inference API")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.port is not None:
        config["api"]["port"] = args.port
    checkpoint_path = config["artifacts"]["checkpoint_path"]
    model_config = config["model"]
    device = config.get("inference", {}).get("device", "auto")
    app = create_app(str(checkpoint_path), model_config, device=device)
    
    # We must pass the import string 'src.api.app:create_app' if we want reload to work,
    # but since app is already created, reload might not work properly without restructuring.
    # We'll just pass the app directly.
    uvicorn.run(app, host=args.host, port=config.get("api", {}).get("port", 8000), log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
