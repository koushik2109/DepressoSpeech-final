from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_rate: int = 8
    chunk_duration: float = 2.0
    chunk_overlap: float = 0.5
    vad_mode: str = "aggressive"
    feature_backends: tuple[str, ...] = ("mfcc", "egemaps", "hubert")
    mfcc_n_mfcc: int = 13
    hubert_model_name: str = "facebook/hubert-base-ls960"


@dataclass(frozen=True)
class VideoConfig:
    frame_rate: int = 8
    temporal_window: int = 32
    openface_csv_key: str = "openface"
    embedding_backend: str = "resnet50"
    encoder_type: str = "transformer"


@dataclass(frozen=True)
class TextConfig:
    transformer_model: str = "sentence-transformers/all-mpnet-base-v2"
    max_tokens: int = 256
    use_session_level: bool = True
    use_chunk_level: bool = True


@dataclass(frozen=True)
class ModelConfig:
    audio_dim: int = 512
    video_dim: int = 512
    text_dim: int = 384
    fusion_dim: int = 256
    num_questions: int = 8
    fusion_mode: str = "hybrid"
    dropout: float = 0.2
    encoder_type: str = "transformer"
    fusion_layers: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    num_workers: int = 2
    max_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 8
    use_amp: bool = True
    modality_dropout: float = 0.2
    temporal_dropout: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class InferenceConfig:
    device: str = "auto"
    batch_size: int = 4
    sanitize_values: bool = True
    allow_missing_modalities: bool = True
    max_batch_items: int = 16


@dataclass(frozen=True)
class PipelineConfig:
    audio: AudioConfig = AudioConfig()
    video: VideoConfig = VideoConfig()
    text: TextConfig = TextConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()


def load_config(config_path: Path) -> Dict[str, Any]:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
