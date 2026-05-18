from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Optional
from src.features.preprocessing import FeaturePreprocessor, PCATransform
import logging
logger = logging.getLogger(__name__)

from src.features.sanitization import sanitize_array


class TextFeatureExtractor:
    """SentenceTransformer-based text embedding for transcripts.

    Optionally applies a serialized `FeaturePreprocessor` (normalizer + PCA)
    saved during training so the runtime embeddings match the model input
    dimensionality (e.g. 768 -> 133).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", preprocessor_path: Optional[Path] = None):
        self.model = SentenceTransformer(model_name)
        self.preprocessor: Optional[FeaturePreprocessor] = None
        if preprocessor_path is not None:
            p = Path(preprocessor_path)
            if p.exists():
                try:
                    self.preprocessor = FeaturePreprocessor.load(p)
                    logger.info("Loaded FeaturePreprocessor from %s", str(p))
                except Exception:
                    try:
                        # Try loading a raw PCA reducer if present
                        pca = PCATransform.load(p)
                        class _PCAWrap:
                            def __init__(self, pca):
                                self._pca = pca
                            def transform(self, x):
                                return self._pca.transform(x)
                        self.preprocessor = _PCAWrap(pca)
                        logger.info("Loaded PCATransform from %s", str(p))
                    except Exception:
                        logger.warning("Failed to load preprocessor from %s; continuing without it", str(p))
                        self.preprocessor = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        arr = np.asarray(embeddings, dtype=np.float32)
        if self.preprocessor is not None:
            try:
                arr = self.preprocessor.transform(arr)
            except Exception:
                pass
        return sanitize_array(arr)

    def encode_text(self, transcript: str, chunks: Sequence[str]) -> Dict[str, np.ndarray]:
        session_embedding = self.encode([transcript])[0]
        chunk_embeddings = self.encode(chunks)
        return {
            "session": session_embedding,
            "chunks": chunk_embeddings,
        }
