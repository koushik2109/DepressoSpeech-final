from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim))
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn = self.proj(x)
        scores = torch.matmul(attn, self.query)
        scores = scores.masked_fill(~mask, float("-1e9"))
        weights = torch.softmax(scores, dim=1)
        weights = weights.masked_fill(~mask, 0.0)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = weights / denom
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class TemporalConvEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(hidden_dim, 1)
        self.input_norm = nn.LayerNorm(input_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        out = self.conv(x).transpose(1, 2)
        out = self.proj(out)
        return out * mask.unsqueeze(-1).float()


class ConvGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        conv_out = self.conv(x).transpose(1, 2)
        lengths = mask.sum(dim=1).cpu()
        lengths_clamped = lengths.clamp(min=1)
        packed = pack_padded_sequence(conv_out, lengths_clamped, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=conv_out.size(1))
        out = self.output_proj(out)
        out = out * mask.unsqueeze(-1).float()
        return out

class BiGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        lengths = mask.sum(dim=1).cpu()
        lengths_clamped = lengths.clamp(min=1)
        packed = pack_padded_sequence(x, lengths_clamped, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        return out * mask.unsqueeze(-1).float()


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.embed_dim = ((input_dim + num_heads - 1) // num_heads) * num_heads
        self.input_proj = nn.Linear(input_dim, self.embed_dim) if self.embed_dim != input_dim else nn.Identity()
        
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=max(self.embed_dim * 2, 128),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(self.embed_dim, input_dim) if self.embed_dim != input_dim else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_proj(x)
        src_key_padding_mask = ~mask
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        out = self.output_proj(out)
        return out * mask.unsqueeze(-1).float()
