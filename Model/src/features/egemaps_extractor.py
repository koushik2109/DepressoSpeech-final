"""
[LAYER_START] Feature Extraction - eGeMAPS Extractor
Extracts eGeMAPSv02 functionals (88-dim) from audio via OpenSMILE.

Both training and inference use the same audio-based extraction path,
ensuring exact feature parity.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Union, List, Sequence

from src.features.constants import EGEMAPS_DIM

logger = logging.getLogger(__name__)

# Minimum segment duration (seconds) for reliable functional computation
_MIN_SEGMENT_SECONDS = 0.25


class EgemapsExtractor:
    """
    eGeMAPS feature extractor using OpenSMILE eGeMAPSv02 Functionals.

    Both training and inference extract 88 functionals from audio segments.
    Training segments are sliced per transcript utterance timestamps.

    Output: (N, 88) array of eGeMAPSv02 functionals per segment
    """

    EXPECTED_DIM = EGEMAPS_DIM
    BEHAVIORAL_REPORT_FEATURES = (
        ("f0_mean", 0),
        ("f0_std", 1),
        ("jitter", 2),
        ("shimmer", 3),
        ("loudness_mean", 4),
        ("loudness_std", 5),
    )

    def __init__(self):
        self._smile = None

    def _init_opensmile(self):
        """Lazy-load OpenSMILE."""
        if self._smile is None:
            try:
                import opensmile
                self._smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )
                logger.info("[LAYER_START] OpenSMILE eGeMAPSv02 Functionals initialized")
            except ImportError:
                raise ImportError(
                    "opensmile is required. Install with: pip install opensmile"
                )

    # =========================================================
    # UNIFIED: Extract from audio segments (training + inference)
    # =========================================================
    def extract_from_audio(
        self,
        audio_segments: Union[np.ndarray, List[np.ndarray], Sequence[np.ndarray]],
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Extract eGeMAPS functionals from audio segments.

        Accepts both fixed-length chunks (np.ndarray of shape (N, samples))
        and variable-length segments (list of 1-D arrays), as produced by
        transcript-based slicing.

        Args:
            audio_segments: Audio data — either (N, samples_per_chunk) ndarray
                            or list of 1-D arrays with variable lengths.
            sr: Sample rate (default 16000)

        Returns:
            np.ndarray of shape (N, 88)
        """
        self._init_opensmile()
        min_samples = int(_MIN_SEGMENT_SECONDS * sr)

        features_list = []
        for i, chunk in enumerate(audio_segments):
            try:
                chunk = np.asarray(chunk, dtype=np.float32).ravel()
                if chunk.shape[0] < min_samples:
                    chunk = np.pad(chunk, (0, min_samples - chunk.shape[0]))

                feat = self._smile.process_signal(chunk, sr)
                feat_array = feat.values.flatten().astype(np.float32)
                feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=0.0, neginf=0.0)

                if np.allclose(feat_array, 0.0) or np.isclose(feat_array.std(), 0.0):
                    logger.warning(
                        f"[VALIDATION_CHECK] Chunk {i}: eGeMAPS vector is near-constant or zero"
                    )

                if feat_array.shape[0] != self.EXPECTED_DIM:
                    logger.warning(
                        f"[VALIDATION_CHECK] Chunk {i}: expected {self.EXPECTED_DIM}, "
                        f"got {feat_array.shape[0]}"
                    )
                    if feat_array.shape[0] > self.EXPECTED_DIM:
                        feat_array = feat_array[:self.EXPECTED_DIM]
                    else:
                        pad_width = self.EXPECTED_DIM - feat_array.shape[0]
                        feat_array = np.pad(feat_array, (0, pad_width), constant_values=0.0)

                features_list.append(feat_array)
            except Exception as e:
                logger.error(f"[DEBUG_POINT] Chunk {i} failed: {e}")
                features_list.append(np.zeros(self.EXPECTED_DIM, dtype=np.float32))

        result = np.stack(features_list, axis=0)
        logger.info(
            f"[DATA_FLOW] eGeMAPS extracted: {result.shape}, "
            f"mean={float(np.mean(result)):.4f}, std={float(np.std(result)):.4f}"
        )
        return result

    # =========================================================
    # UNIFIED INTERFACE
    # =========================================================
    def extract(
        self,
        audio_segments: Union[np.ndarray, List[np.ndarray], None] = None,
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Extract eGeMAPS functionals from audio segments.

        Args:
            audio_segments: Audio data (fixed or variable-length chunks)
            sr: Sample rate
        """
        if audio_segments is not None:
            return self.extract_from_audio(audio_segments, sr)
        else:
            raise ValueError("Must provide audio_segments")
