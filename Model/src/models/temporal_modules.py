"""Improved temporal modeling architectures optimized for small datasets.

These architectures prioritize:
- Generalization with limited data
- Stable convergence
- Parameter efficiency
- Temporal coherence
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LightweightTemporalCNN(nn.Module):
    """Lightweight temporal CNN for small datasets.

    Uses depthwise-separable convolutions to reduce parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)

        layers = []
        in_channels = input_dim
        out_channels = hidden_dim

        for i in range(num_layers):
            # Depthwise-separable convolution
            layers.append(nn.Sequential(
                # Depthwise: apply 1 filter per input channel
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=in_channels,
                    bias=False,
                ),
                # Pointwise: 1x1 convolution
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            in_channels = out_channels

        self.conv_stack = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Process temporal features.

        Args:
            x: [batch_size, time_steps, input_dim]
            mask: [batch_size, time_steps]

        Returns:
            [batch_size, time_steps, hidden_dim]
        """
        x = self.input_norm(x)
        x = x.transpose(1, 2)  # [batch, input_dim, time]
        x = self.conv_stack(x)
        x = x.transpose(1, 2)  # [batch, time, hidden_dim]
        x = self.output_proj(x)
        return x * mask.unsqueeze(-1).float()


class StableGRUEncoder(nn.Module):
    """Stable GRU encoder for small datasets.

    Uses layer normalization and careful initialization for better convergence.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Projection for bidirectional output
        if bidirectional:
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.output_proj = nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Process temporal features.

        Args:
            x: [batch_size, time_steps, input_dim]
            mask: [batch_size, time_steps]

        Returns:
            [batch_size, time_steps, hidden_dim]
        """
        x = self.input_norm(x)
        lengths = mask.sum(dim=1).cpu()
        lengths_clamped = lengths.clamp(min=1)

        # Pack for efficient RNN processing
        packed = pack_padded_sequence(x, lengths_clamped, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        # Project if bidirectional
        out = self.output_proj(out)

        # Apply mask
        return out * mask.unsqueeze(-1).float()


class HybridTemporalModule(nn.Module):
    """Hybrid CNN+GRU for optimal temporal modeling.

    Combines CNN for local pattern detection with GRU for long-range dependencies.
    Optimized for small datasets.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)

        # Light CNN for local patterns
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # GRU for temporal context
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Process temporal features.

        Args:
            x: [batch_size, time_steps, input_dim]
            mask: [batch_size, time_steps]

        Returns:
            [batch_size, time_steps, hidden_dim]
        """
        x = self.input_norm(x)

        # CNN: local pattern extraction
        x_t = x.transpose(1, 2)  # [batch, input_dim, time]
        x_conv = self.conv(x_t).transpose(1, 2)  # [batch, time, hidden_dim]

        # GRU: temporal context
        lengths = mask.sum(dim=1).cpu()
        lengths_clamped = lengths.clamp(min=1)
        packed = pack_padded_sequence(x_conv, lengths_clamped, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        x_gru, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        # Project and mask
        out = self.proj(x_gru)
        return out * mask.unsqueeze(-1).float()


class AttentionPooling(nn.Module):
    """Learnable attention pooling for temporal aggregation.

    Better than mean pooling for capturing important temporal elements.
    """

    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feature_dim))
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling.

        Args:
            x: [batch_size, time_steps, feature_dim]
            mask: [batch_size, time_steps]

        Returns:
            Pooled output [batch_size, feature_dim]
        """
        # Compute attention scores
        attn = self.proj(x)  # [batch, time, feature_dim]
        scores = torch.matmul(attn, self.query)  # [batch, time]

        # Apply mask
        scores = scores.masked_fill(~mask, float("-1e9"))

        # Softmax
        weights = torch.softmax(scores, dim=1)  # [batch, time]

        # Weighted sum
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # [batch, feature_dim]


class StatisticsPooling(nn.Module):
    """Statistics-based temporal pooling (mean + std).

    Preserves information about feature variation which can be important
    for behavioral signals.
    """

    def __init__(self, feature_dim: int):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute statistics over time.

        Args:
            x: [batch_size, time_steps, feature_dim]
            mask: [batch_size, time_steps]

        Returns:
            Concatenated mean and std [batch_size, 2 * feature_dim]
        """
        # Apply mask
        x_masked = x * mask.unsqueeze(-1).float()

        # Compute mean
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = x_masked.sum(dim=1) / lengths  # [batch, feature_dim]

        # Compute std
        x_centered = (x - mean.unsqueeze(1)) * mask.unsqueeze(-1).float()
        std = torch.sqrt((x_centered ** 2).sum(dim=1) / lengths + 1e-8)  # [batch, feature_dim]

        # Concatenate
        return torch.cat([mean, std], dim=-1)  # [batch, 2 * feature_dim]


class TemporalAttentionAggregation(nn.Module):
    """Multi-head attention for temporal aggregation.

    More expressive than single-head attention but still parameter-efficient.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply temporal self-attention.

        Args:
            x: [batch_size, time_steps, feature_dim]
            mask: [batch_size, time_steps]

        Returns:
            Aggregated output [batch_size, feature_dim]
        """
        # Self-attention
        src_key_padding_mask = ~mask
        attn_out, _ = self.attention(x, x, x, key_padding_mask=src_key_padding_mask)
        attn_out = self.norm(attn_out + x)

        # Mean pooling
        return (attn_out * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
