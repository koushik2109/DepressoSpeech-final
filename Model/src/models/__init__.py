"""
Model Architecture — DepressoSpeech

Supports:
  - DepressionModel: Single-modality (audio or text) with MLP/BiGRU/Attention
  - MultimodalFusion: Audio+Text bimodal fusion (V1)
  - GatedMultimodalModel: Gated bimodal fusion with quality gates
  - TrimodalFusionModel: Audio+Video+Text with cross-modal attention (V2)
  - VideoEncoder: OpenFace+CNN video feature encoder
"""
from .mlp_block import MLPBlock
from .bigru import BiGRUEncoder
from .attention import AttentionPooling
from .statistics_pooling import StatisticsPooling
from .depression_model import DepressionModel
from .multimodal_fusion import MultimodalFusion, AudioEncoder
from .gated_fusion_model import GatedMultimodalModel
from .video_encoder import VideoEncoder
from .multimodal_fusion_v2 import TrimodalFusionModel, CrossModalAttention, FusionOutput

__all__ = [
    "MLPBlock",
    "BiGRUEncoder",
    "AttentionPooling",
    "StatisticsPooling",
    "DepressionModel",
    "MultimodalFusion",
    "AudioEncoder",
    "GatedMultimodalModel",
    "VideoEncoder",
    "TrimodalFusionModel",
    "CrossModalAttention",
    "FusionOutput",
]
