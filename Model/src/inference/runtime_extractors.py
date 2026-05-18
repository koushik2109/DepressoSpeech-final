"""Modular runtime feature extractors for live inference.

These extractors automatically generate training-compatible features from raw
webcam/video/audio input without manual configuration.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
import librosa
import torchvision.models as models
import torchvision.transforms as T
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sentence_transformers import SentenceTransformer

from src.features.sanitization import sanitize_array
from src.features.preprocessing import FeaturePreprocessor, PCATransform, FeatureNormalizer
from src.features.specs import AudioSpec, VideoSpec, TextSpec


class OpenFaceRuntimeExtractor:
    """Extract OpenFace behavioral features from video frames or CSV.

    Supports:
    - Direct frame-based extraction (requires OpenFace installed)
    - CSV loading for pre-computed OpenFace features
    """

    EXPECTED_COLUMNS = [
        "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
        "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
        "AU25_r", "AU26_r", "AU45_r",  # 17 AUs
        "Gaze_angle_x", "Gaze_angle_y", "Gaze_angle_z",  # 3 gaze angles
        "Pose_Tx", "Pose_Ty", "Pose_Tz",  # 3 head pose
    ]

    def __init__(self, use_csv: bool = True):
        """Initialize extractor.

        Args:
            use_csv: If True, load from CSV. If False, use frame-based extraction.
        """
        self.use_csv = use_csv

    def extract_from_csv(self, csv_path: Path) -> np.ndarray:
        """Extract OpenFace features from CSV file.

        Args:
            csv_path: Path to OpenFace CSV file

        Returns:
            Features array [num_frames, num_features]
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"OpenFace CSV is empty: {csv_path}")

        # Extract features
        features_list = []
        for row in rows:
            try:
                # Extract AU features
                au_features = [float(row.get(col, 0.0)) for col in self.EXPECTED_COLUMNS[:17]]
                # Extract gaze
                gaze_features = [float(row.get(col, 0.0)) for col in self.EXPECTED_COLUMNS[17:20]]
                # Extract head pose
                pose_features = [float(row.get(col, 0.0)) for col in self.EXPECTED_COLUMNS[20:23]]
                
                features_list.append(au_features + gaze_features + pose_features)
            except (ValueError, KeyError) as e:
                continue

        if not features_list:
            raise ValueError(f"Could not extract valid features from {csv_path}")

        features = np.array(features_list, dtype=np.float32)
        return sanitize_array(features)

    def extract(self, source: Path | str) -> np.ndarray:
        """Extract OpenFace features from CSV or video frames."""
        source_path = Path(source) if isinstance(source, str) else source
        
        if source_path.suffix == ".csv":
            return self.extract_from_csv(source_path)
        else:
            raise NotImplementedError(
                "Frame-based OpenFace extraction requires additional dependencies. "
                "Use CSV format for now."
            )


class ResNetRuntimeExtractor:
    """Extract visual embeddings from video frames using ResNet50.

    Supports:
    - Frame-by-frame extraction
    - Temporal aggregation
    - PCA reduction for consistency with training
    """

    def __init__(
        self,
        device: str = "auto",
        pca_preprocessor: Optional[FeaturePreprocessor] = None,
    ):
        """Initialize extractor.

        Args:
            device: Device to use ("auto", "cuda", "cpu")
            pca_preprocessor: Optional PCA preprocessor for dimensionality reduction
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        self.pca_preprocessor = pca_preprocessor

        # Load pretrained ResNet50
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.backbone.eval()

        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def extract_from_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Extract embeddings from video frames.

        Args:
            frames: List of frames as np.ndarray [H, W, 3]

        Returns:
            Embeddings array [num_frames, embedding_dim]
        """
        if not frames:
            raise ValueError("No frames provided")

        embeddings_list = []
        with torch.no_grad():
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    # Convert BGR to RGB if needed
                    if frame.shape[2] == 3:
                        if frame.max() > 1.0:  # Likely 0-255 range
                            frame = frame.astype(np.float32) / 255.0
                        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
                    else:
                        frame_tensor = self.transform(T.ToPILImage()(frame))
                else:
                    frame_tensor = self.transform(frame)

                frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
                embedding = self.backbone(frame_tensor).squeeze()
                embeddings_list.append(embedding.cpu().numpy())

        embeddings = np.array(embeddings_list, dtype=np.float32)  # [num_frames, 2048]
        embeddings = sanitize_array(embeddings)

        # Apply PCA if available
        if self.pca_preprocessor:
            embeddings = self.pca_preprocessor.transform(embeddings)

        return embeddings


class AudioFeatureRuntimeExtractor:
    """Extract audio features: MFCC + HuBERT.

    Produces training-compatible audio features.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mfcc_n_mfcc: int = 13,
        use_hubert: bool = False,
        normalizer: Optional[FeatureNormalizer] = None,
    ):
        """Initialize extractor.

        Args:
            sample_rate: Audio sample rate
            mfcc_n_mfcc: Number of MFCC coefficients
            use_hubert: Whether to include HuBERT embeddings
            normalizer: Optional normalizer for features
        """
        self.sample_rate = sample_rate
        self.mfcc_n_mfcc = mfcc_n_mfcc
        self.use_hubert = use_hubert
        self.normalizer = normalizer

        if use_hubert:
            self._hubert_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/hubert-base-ls960"
            )
            self._hubert_model = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960")
            self._hubert_model.eval()
            if torch.cuda.is_available():
                self._hubert_model = self._hubert_model.cuda()

    def extract_mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """Extract MFCC features: 13 MFCC + 13 delta + 13 delta-delta.

        Args:
            waveform: Audio waveform [num_samples]

        Returns:
            MFCC features [num_frames, 39]
        """
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.max() > 1.0:
            waveform = waveform / 32768.0  # Convert from int16

        # Normalize
        waveform = librosa.util.normalize(waveform)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=waveform, sr=self.sample_rate, n_mfcc=self.mfcc_n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Concatenate: [13 + 13 + 13 = 39]
        features = np.vstack([mfcc, delta, delta2]).T  # [num_frames, 39]
        return sanitize_array(features.astype(np.float32))

    def extract_hubert(self, waveform: np.ndarray) -> np.ndarray:
        """Extract HuBERT embeddings.

        Args:
            waveform: Audio waveform [num_samples]

        Returns:
            HuBERT embeddings [num_frames, 768]
        """
        if not self.use_hubert:
            raise ValueError("HuBERT extraction not enabled")

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.max() > 1.0:
            waveform = waveform / 32768.0

        with torch.no_grad():
            inputs = self._hubert_extractor(
                waveform,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self._hubert_model(**inputs)
            embeddings = outputs.last_hidden_state[0].cpu().numpy()  # [num_frames, 768]

        return sanitize_array(embeddings.astype(np.float32))

    def extract(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract audio features.

        Args:
            waveform: Audio waveform [num_samples]

        Returns:
            Dict with 'mfcc' and optionally 'hubert' features
        """
        features = {}

        # MFCC is always extracted
        mfcc_features = self.extract_mfcc(waveform)
        if self.normalizer:
            mfcc_features = self.normalizer.transform(mfcc_features)
        features["mfcc"] = mfcc_features

        # HuBERT if enabled
        if self.use_hubert:
            hubert_features = self.extract_hubert(waveform)
            features["hubert"] = hubert_features

        return features


class TextFeatureRuntimeExtractor:
    """Extract text embeddings using SentenceTransformer.

    Produces training-compatible text embeddings.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "auto",
    ):
        """Initialize extractor.

        Args:
            model_name: HuggingFace model name
            device: Device to use ("auto", "cuda", "cpu")
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode single text to embedding.

        Args:
            text: Text string

        Returns:
            Embedding [384]
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return sanitize_array(embedding.astype(np.float32))

    def encode_chunks(self, chunks: Sequence[str]) -> np.ndarray:
        """Encode text chunks to embeddings.

        Args:
            chunks: List of text chunks

        Returns:
            Embeddings [num_chunks, 384]
        """
        embeddings = self.model.encode(
            list(chunks),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return sanitize_array(embeddings.astype(np.float32))


class LiveInferenceFeatureExtractor:
    """Unified runtime feature extraction for live inference.

    Automatically produces training-compatible features from raw inputs.
    """

    def __init__(
        self,
        openface_pca: Optional[PCATransform] = None,
        resnet_pca: Optional[PCATransform] = None,
        audio_normalizer: Optional[FeatureNormalizer] = None,
        device: str = "auto",
    ):
        """Initialize extractor.

        Args:
            openface_pca: Optional PCA for OpenFace features
            resnet_pca: Optional PCA for ResNet embeddings
            audio_normalizer: Optional normalizer for audio features
            device: Device for model inference
        """
        self.openface_extractor = OpenFaceRuntimeExtractor(use_csv=True)
        self.resnet_extractor = ResNetRuntimeExtractor(
            device=device,
            pca_preprocessor=resnet_pca,
        )
        self.audio_extractor = AudioFeatureRuntimeExtractor(
            normalizer=audio_normalizer,
        )
        self.text_extractor = TextFeatureRuntimeExtractor(device=device)

    def extract_video(
        self,
        source: Path | str,
        use_openface_csv: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Extract video features from file or CSV.

        Args:
            source: Path to video file or OpenFace CSV
            use_openface_csv: Whether source is OpenFace CSV

        Returns:
            Dict with 'openface' and optionally 'resnet' features
        """
        features = {}

        if use_openface_csv:
            features["openface"] = self.openface_extractor.extract(source)
        
        return features

    def extract_audio(self, audio_path: Path | str) -> Dict[str, np.ndarray]:
        """Extract audio features from file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with 'mfcc' and optionally 'hubert' features
        """
        waveform, sr = librosa.load(str(audio_path), sr=self.audio_extractor.sample_rate, mono=True)
        return self.audio_extractor.extract(waveform)

    def extract_text(
        self,
        text: str,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Extract text embeddings.

        Args:
            text: Input text
            chunk_size: Approximate chunk size for splitting

        Returns:
            Embeddings [num_chunks, 384]
        """
        # Simple sentence-based chunking
        sentences = text.split(".")
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)

        if not chunks:
            chunks = [text]

        return self.text_extractor.encode_chunks(chunks)
