from __future__ import annotations

import logging
import pickle
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager

from src.api.routes import router, register_inferencer, get_inferencer
from src.features.text_features import TextFeatureExtractor
from src.inference.inferencer import ModelV2Inferencer
from src.utils.logging import configure_logging

logger = logging.getLogger(__name__)


async def _warmup_model() -> None:
    """Run a dummy forward pass so PyTorch JIT compiles on startup."""
    inferencer = get_inferencer()
    if inferencer is None:
        return
    try:
        dummy = {
            "audio": [[0.0] * 33],
            "audio_mask": [True],
        }
        inferencer.predict_single(dummy)
        logger.info("Model warmup completed successfully.")
    except Exception as exc:
        logger.warning("Model warmup failed (non-fatal): %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await _warmup_model()
    yield


def _load_preprocessors(checkpoint_path: str) -> dict:
    """Load normalizer+PCA preprocessors for each modality from the training checkpoint."""
    checkpoint_dir = Path(checkpoint_path).parent
    # Search in multimodal_v3 first, then generic preprocessors dir
    candidates = [
        checkpoint_dir / "multimodal_v4" / "preprocessors",
        checkpoint_dir / "multimodal_v3" / "preprocessors",
        checkpoint_dir / "preprocessors",
    ]
    preproc_dir = next((d for d in candidates if d.exists()), None)
    if preproc_dir is None:
        logger.warning("No preprocessors directory found under %s", checkpoint_dir)
        return {}
    preprocessors = {}
    for modality in ("audio", "video", "text"):
        pkl_path = preproc_dir / f"{modality}_preprocessor.pkl"
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    preprocessors[modality] = pickle.load(f)
                logger.info("Loaded %s preprocessor from %s", modality, pkl_path)
            except Exception as exc:
                logger.warning("Failed to load %s preprocessor: %s", modality, exc)
    return preprocessors


def create_app(checkpoint_path: str, model_config: dict, device: str = "auto", text_model_name: str = "sentence-transformers/all-mpnet-base-v2") -> FastAPI:
    configure_logging(Path("logs"), level="INFO")
    inferencer = ModelV2Inferencer(Path(checkpoint_path), model_config, device=device)
    # Attempt to load text PCA/reducer saved during training. Prefer explicit
    # PCA reducer file if present, otherwise fall back to the generic scalers.
    scalers_dir = Path(checkpoint_path).parent / "scalers"
    pca_path = scalers_dir / "pca_reducer.pkl"
    feature_scalers_path = scalers_dir / "feature_scalers.pkl"
    preproc_path = pca_path if pca_path.exists() else (feature_scalers_path if feature_scalers_path.exists() else None)
    text_extractor = TextFeatureExtractor(text_model_name, preprocessor_path=preproc_path)
    preprocessors = _load_preprocessors(checkpoint_path)
    register_inferencer(inferencer, text_extractor=text_extractor, preprocessors=preprocessors)

    app = FastAPI(
        title="ModelV2 Depression Assessment API",
        description="Production-grade FastAPI inference for multimodal depression assessment.",
        version="2.0.0",
        lifespan=_lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="")
    return app
