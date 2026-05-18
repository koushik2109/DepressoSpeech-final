import logging
from pathlib import Path
from datetime import datetime


def configure_logging(log_dir: Path, level: str = "INFO", experiment_name: str = "trimodal") -> Path:
    runs_dir = log_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Prune old runs to keep only the last 10
    import shutil
    existing_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
    if len(existing_runs) >= 10:
        for old_dir in existing_runs[:-9]:  # keep 9 existing + 1 new = 10
            shutil.rmtree(old_dir, ignore_errors=True)
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    file_handler = logging.FileHandler(run_dir / "training.log")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return run_dir
