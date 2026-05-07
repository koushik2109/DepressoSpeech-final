"""
Video Feature Encoder for Multimodal Depression Detection.

Accepts pre-extracted video features:
    - OpenFace:  pose (6D), gaze (8D), AUs (35D) → 49D per frame
    - CNN embed: ResNet/VGG/DenseNet embedding → 512/2048D per frame

Architecture:
    OpenFace: (B, T, 49)   → Linear(49, 64)   → LayerNorm → ReLU
    CNN:      (B, T, 512)  → Linear(512, 64)   → LayerNorm → ReLU
    Concat:   (B, T, 128)  → Conv1D temporal    → StatsPool → (B, 256)
"""

import torch
import torch.nn as nn
import logging

from src.models.statistics_pooling import StatisticsPooling

logger = logging.getLogger(__name__)

# Default dimensions for pre-extracted features
OPENFACE_DIM = 49   # pose(6) + gaze(8) + AUs(35)
CNN_EMBED_DIM = 512  # ResNet18/34 → 512, ResNet50/VGG → 2048


class VideoEncoder(nn.Module):
    """
    Lightweight encoder for pre-extracted video features.

    Processes OpenFace behavioral features and CNN visual embeddings
    through separate projection heads, then combines via temporal
    convolution and statistics pooling.

    ~35K params. No raw video processing — works on pre-extracted features.
    """

    def __init__(
        self,
        openface_dim: int = OPENFACE_DIM,
        cnn_dim: int = CNN_EMBED_DIM,
        proj_dim: int = 64,
        out_dim: int = 128,
        stats_mode: str = "mean_std",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.openface_dim = openface_dim
        self.cnn_dim = cnn_dim

        # OpenFace projection: pose + gaze + AUs → proj_dim
        self.openface_proj = nn.Sequential(
            nn.Linear(openface_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # CNN embedding projection: ResNet/VGG/DenseNet → proj_dim
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal modeling on concatenated features
        concat_dim = proj_dim * 2  # 128
        self.temporal = nn.Sequential(
            nn.Conv1d(concat_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Statistics pooling
        self.pool = StatisticsPooling(input_dim=out_dim, stats=stats_mode)
        self.norm = nn.LayerNorm(self.pool.output_dim)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"VideoEncoder: openface({openface_dim}) + cnn({cnn_dim}) "
            f"→ proj({proj_dim}x2) → conv({out_dim}) → pool({self.pool.output_dim}), "
            f"params={total_params:,}"
        )

    @property
    def output_dim(self) -> int:
        return self.pool.output_dim

    def forward(
        self,
        openface_features: torch.Tensor,
        cnn_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            openface_features: (B, T, openface_dim) — pose+gaze+AU features
            cnn_features:      (B, T, cnn_dim)      — CNN visual embeddings
            mask:              (B, T)                — True for valid frames

        Returns:
            (B, pool_output_dim) — pooled video representation
        """
        # Project each stream
        of_proj = self.openface_proj(openface_features)   # (B, T, 64)
        cnn_proj = self.cnn_proj(cnn_features)            # (B, T, 64)

        # Concatenate
        combined = torch.cat([of_proj, cnn_proj], dim=-1)  # (B, T, 128)

        # Temporal conv: (B, T, C) → (B, C, T) → conv → (B, C, T) → (B, T, C)
        h = combined.transpose(1, 2)
        h = self.temporal(h)
        h = h.transpose(1, 2)

        # Pool + normalize
        pooled = self.pool(h, mask)
        return self.norm(pooled)

    def forward_single_stream(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        stream: str = "openface",
    ) -> torch.Tensor:
        """Forward pass using only one video stream (for missing modality handling).

        Args:
            features: (B, T, D) — features from one stream
            mask: (B, T)
            stream: "openface" or "cnn"

        Returns:
            (B, pool_output_dim)
        """
        B, T, _ = features.shape

        if stream == "openface":
            proj = self.openface_proj(features)
            pad = torch.zeros(B, T, proj.shape[-1], device=features.device)
            combined = torch.cat([proj, pad], dim=-1)
        else:
            proj = self.cnn_proj(features)
            pad = torch.zeros(B, T, proj.shape[-1], device=features.device)
            combined = torch.cat([pad, proj], dim=-1)

        h = combined.transpose(1, 2)
        h = self.temporal(h)
        h = h.transpose(1, 2)

        pooled = self.pool(h, mask)
        return self.norm(pooled)
