"""
[LAYER_START] Session 9: FastAPI Application
REST API for depression severity prediction from audio.

[INFERENCE_PATH] HTTP request → InferencePipeline → JSON response.

Startup: loads inference pipeline with trained model + artifacts.
Shutdown: releases resources.
"""

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from src.inference.pipeline import InferencePipeline
from src.inference.fusion_pipeline import FusionInferencePipeline
from src.utils.experiment_tracker import ExperimentTracker

# --- Rate limiter (SEC-3) ---
limiter = Limiter(key_func=get_remote_address)

# --- API key auth (SEC-2) ---
# Set via environment variable; if unset, auth is disabled (dev mode)
API_KEY = os.environ.get("DEPRESSO_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(_api_key_header)) -> Optional[str]:
    """Validate API key if authentication is enabled."""
    if not API_KEY:
        return None  # Auth disabled (no key configured)
    if not api_key or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

logger = logging.getLogger(__name__)

# Module-level references, set during lifespan
_pipeline: Optional[InferencePipeline] = None
_multimodal_predictor = None  # Optional MultimodalPredictor
_tracker: Optional[ExperimentTracker] = None
_pipeline_lock = threading.Lock()  # BP-1: Thread safety for concurrent requests

# Default config path
DEFAULT_CONFIG_PATH = "configs/inference_config.yaml"


def _load_config(config_path: str) -> dict:
    """Load inference config from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"[API] Config not found: {path}, using defaults")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _init_pipeline(config: dict):
    """Initialize the inference pipeline from config. Supports fusion or text-only mode."""
    artifacts = config.get("artifacts", {})
    inference_cfg = config.get("inference", {})
    pipeline_mode = config.get("pipeline_mode", "fusion")

    if pipeline_mode == "fusion":
        logger.info("[API] Initializing FUSION pipeline (text + audio + behavioral)")
        return FusionInferencePipeline(
            fusion_checkpoint=artifacts.get("fusion_checkpoint", "checkpoints/best_fusion.pt"),
            text_checkpoint=artifacts.get("text_checkpoint", "checkpoints/best_model.pt"),
            # Fusion preprocessor expects a full preprocessing config tree
            # (audio + preprocessing + paths), not only the audio subsection.
            audio_config=config,
            device=inference_cfg.get("device", "auto"),
            use_text_transcription=inference_cfg.get("use_text_transcription", False),
        )
    else:
        logger.info("[API] Initializing TEXT-ONLY pipeline (legacy)")
        model_cfg = config.get("model", {})
        return InferencePipeline(
            model_path=artifacts.get("model_path", "checkpoints/best_model.pt"),
            normalizer_path=artifacts.get("normalizer_path"),
            pca_path=artifacts.get("pca_path"),
            audio_config=config.get("audio"),
            model_config=model_cfg or None,
            device=inference_cfg.get("device", "auto"),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model + DB. Shutdown: release resources."""
    global _pipeline, _tracker, _multimodal_predictor

    config_path = getattr(app.state, "config_path", DEFAULT_CONFIG_PATH)
    logger.info(f"[API] Loading config from: {config_path}")

    config = _load_config(config_path)
    _pipeline = _init_pipeline(config)
    logger.info("[API] Inference pipeline loaded — ready to serve")

    # Initialize multimodal predictor (optional — graceful if fails)
    try:
        from src.inference.multimodal_pipeline import MultimodalPredictor
        ckpt_path = config.get("artifacts", {}).get(
            "trimodal_checkpoint", "checkpoints/trimodal_fusion.pt"
        )
        device = config.get("inference", {}).get("device", "auto")
        _multimodal_predictor = MultimodalPredictor(
            checkpoint_path=ckpt_path, device=device
        )
        logger.info("[API] MultimodalPredictor initialized")
    except Exception as e:
        logger.warning(f"[API] MultimodalPredictor init failed (will use fallback): {e}")
        _multimodal_predictor = None

    # Initialize DB tracker (creates tables if needed)
    try:
        db_config_path = config.get("api", {}).get(
            "db_config", "configs/db_config.yaml"
        )
        _tracker = ExperimentTracker(db_config_path=db_config_path)
        logger.info("[API] Database tracker initialized")
    except Exception as e:
        logger.warning(f"[API] DB tracker init failed (predictions won't be logged): {e}")
        _tracker = None

    yield

    _pipeline = None
    _multimodal_predictor = None
    _tracker = None
    logger.info("[API] Shutdown complete")


def create_app(config_path: str = DEFAULT_CONFIG_PATH) -> FastAPI:
    """
    Factory function to create the FastAPI application.

    Args:
        config_path: Path to inference_config.yaml

    Returns:
        Configured FastAPI instance
    """
    config = _load_config(config_path)

    app = FastAPI(
        title="DepressoSpeech API",
        description=(
            "Depression severity prediction from audio using "
            "eGeMAPS + MFCC + SBERT features with MLP + BiGRU + Attention model. "
            "Predicts PHQ-8 scores (0-24) with clinical severity labels."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store config path for lifespan to pick up
    app.state.config_path = config_path

    # --- Rate limiter (SEC-3) ---
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )

    # GZip — compress JSON prediction responses over 1KB
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # CORS middleware — origins loaded from config; restrict in production
    cors_origins = config.get("api", {}).get("cors_origins", ["*"])
    allow_credentials = "*" not in cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes — pass API config for limits
    from src.api import routes

    api_cfg = config.get("api", {})
    if api_cfg.get("max_file_size_mb"):
        routes.MAX_FILE_SIZE_BYTES = api_cfg["max_file_size_mb"] * 1024 * 1024
    if api_cfg.get("max_batch_size"):
        routes.MAX_BATCH_SIZE = api_cfg["max_batch_size"]
    if api_cfg.get("allowed_extensions"):
        routes.ALLOWED_EXTENSIONS = set(api_cfg["allowed_extensions"])

    app.include_router(routes.router)

    return app
