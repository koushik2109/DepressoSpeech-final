from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from src.models.encoders import (
    AttentionPooling,
    BiGRUEncoder,
    ConvGRUEncoder,
    TemporalConvEncoder,
    TransformerEncoder,
)
from src.models.fusion import MultimodalFusion
from src.models.heads import ClassificationHead, ConfidenceHead, QuestionHead, RegressionHead


class MultimodalDepressionModel(nn.Module):
    def __init__(
        self,
        audio_dim: int = 512,
        video_dim: int = 512,
        text_dim: int = 384,
        fusion_dim: int = 256,
        num_questions: int = 8,
        encoder_type: str = "transformer",
        pooling_type: str = "attention",
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        fusion_mode: str = "hybrid",
        modality_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.pooling_type = pooling_type
        self.audio_input_dim = audio_dim
        self.video_input_dim = video_dim
        self.text_input_dim = text_dim

        self.audio_encoder = self._build_encoder(audio_dim, fusion_dim, dropout, num_layers, num_heads)
        self.video_encoder = self._build_encoder(video_dim, fusion_dim, dropout, num_layers, num_heads)
        self.text_encoder = self._build_encoder(text_dim, fusion_dim, dropout, num_layers, num_heads)

        self.audio_projection = self._build_projection(audio_dim, fusion_dim)
        self.video_projection = self._build_projection(video_dim, fusion_dim)
        self.text_projection = self._build_projection(text_dim, fusion_dim)

        self.audio_missing = nn.Parameter(torch.zeros(fusion_dim))
        self.video_missing = nn.Parameter(torch.zeros(fusion_dim))
        self.text_missing = nn.Parameter(torch.zeros(fusion_dim))

        # Persistent attention poolers — one per modality so weights ARE trained.
        # (previously created fresh each forward pass → random, never trained → CCC gap)
        if self.pooling_type == "attention":
            self.audio_pooler = AttentionPooling(fusion_dim)
            self.video_pooler = AttentionPooling(fusion_dim)
            self.text_pooler  = AttentionPooling(fusion_dim)

        self.fusion = MultimodalFusion(
            embed_dim=fusion_dim,
            fusion_dim=fusion_dim,
            mode=fusion_mode,
            dropout=dropout,
        )
        self.phq_head = RegressionHead(fusion_dim)
        self.question_head = QuestionHead(fusion_dim, num_questions)
        self.classification_head = ClassificationHead(fusion_dim)
        self.confidence_head = ConfidenceHead(fusion_dim)

        self.modality_dropout = modality_dropout

    def _build_encoder(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        num_heads: int,
    ) -> nn.Module:
        if self.encoder_type == "conv":
            return TemporalConvEncoder(input_dim, hidden_dim, dropout=dropout)
        if self.encoder_type == "conv_gru":
            return ConvGRUEncoder(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        if self.encoder_type == "bigru":
            return BiGRUEncoder(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        return TransformerEncoder(input_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout)

    def _build_projection(self, input_dim: int, hidden_dim: int) -> nn.Module:
        if self.encoder_type == "bigru":
            return nn.Linear(hidden_dim * 2, hidden_dim)
        if self.encoder_type == "transformer":
            return nn.Linear(input_dim, hidden_dim)
        return nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def _mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        length = mask.sum(dim=1).clamp(min=1.0)
        return summed / length

    def _pooled_representation(
        self, x: torch.Tensor, mask: torch.Tensor, pooler: Optional[nn.Module] = None
    ) -> torch.Tensor:
        if self.pooling_type == "attention" and pooler is not None:
            return pooler(x, mask)
        return self._mean_pool(x, mask)

    def _encode_modality(
        self,
        x: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        encoder: nn.Module,
        projector: nn.Module,
        missing_token: nn.Parameter,
        pooler: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x is None or x.numel() == 0 or mask is None:
            batch = x.shape[0] if x is not None else 1
            return missing_token.unsqueeze(0).expand(batch, -1), torch.zeros(batch, dtype=torch.bool, device=missing_token.device)

        mask = mask.to(dtype=torch.bool)
        present = mask.any(dim=1)
        encoded = encoder(x, mask)
        pooled = self._pooled_representation(encoded, mask, pooler)
        projected = projector(pooled)

        if not present.all():
            fallback = missing_token.unsqueeze(0).expand(projected.size(0), -1)
            projected = projected * present.unsqueeze(-1).to(projected.dtype) + fallback * (~present).unsqueeze(-1).to(projected.dtype)

        return projected, present

    def _drop_modality(
        self,
        modality_repr: torch.Tensor,
        present: torch.Tensor,
        missing_token: nn.Parameter,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0.0:
            return modality_repr, present

        keep = torch.rand(modality_repr.size(0), device=modality_repr.device) > self.modality_dropout
        keep = keep & present
        if keep.all():
            return modality_repr, present

        fallback = missing_token.unsqueeze(0).expand_as(modality_repr)
        modality_repr = modality_repr * keep.unsqueeze(-1).to(modality_repr.dtype) + fallback * (~keep).unsqueeze(-1).to(modality_repr.dtype)
        return modality_repr, keep

    def forward(
        self,
        audio: Optional[torch.Tensor],
        video: Optional[torch.Tensor],
        text: Optional[torch.Tensor],
        audio_mask: Optional[torch.Tensor],
        video_mask: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        audio_repr, audio_present = self._encode_modality(
            audio,
            audio_mask,
            self.audio_encoder,
            self.audio_projection,
            self.audio_missing,
            pooler=getattr(self, "audio_pooler", None),
        )
        video_repr, video_present = self._encode_modality(
            video,
            video_mask,
            self.video_encoder,
            self.video_projection,
            self.video_missing,
            pooler=getattr(self, "video_pooler", None),
        )
        text_repr, text_present = self._encode_modality(
            text,
            text_mask,
            self.text_encoder,
            self.text_projection,
            self.text_missing,
            pooler=getattr(self, "text_pooler", None),
        )

        if self.training and self.modality_dropout > 0.0:
            a_keep = torch.rand(audio_repr.size(0), device=audio_repr.device) > self.modality_dropout
            v_keep = torch.rand(video_repr.size(0), device=video_repr.device) > self.modality_dropout
            t_keep = torch.rand(text_repr.size(0), device=text_repr.device) > self.modality_dropout
            
            # If all are false, force one to be true among the ones that are actually present
            a_keep = a_keep & audio_present
            v_keep = v_keep & video_present
            t_keep = t_keep & text_present
            all_dropped = ~(a_keep | v_keep | t_keep)
            
            # For rows where all dropped, try text, then audio, then video
            if all_dropped.any():
                t_keep = t_keep | (all_dropped & text_present)
                all_dropped_still = all_dropped & ~text_present
                a_keep = a_keep | (all_dropped_still & audio_present)
                all_dropped_still = all_dropped_still & ~audio_present
                v_keep = v_keep | (all_dropped_still & video_present)
            
            def apply_drop(repr_t, present_t, keep_t, missing_t):
                if keep_t.all(): return repr_t, present_t
                fallback = missing_t.unsqueeze(0).expand_as(repr_t)
                repr_out = repr_t * keep_t.unsqueeze(-1).to(repr_t.dtype) + fallback * (~keep_t).unsqueeze(-1).to(repr_t.dtype)
                return repr_out, keep_t
                
            audio_repr, audio_present = apply_drop(audio_repr, audio_present, a_keep, self.audio_missing)
            video_repr, video_present = apply_drop(video_repr, video_present, v_keep, self.video_missing)
            text_repr, text_present = apply_drop(text_repr, text_present, t_keep, self.text_missing)

        modality_present = torch.stack([audio_present, video_present, text_present], dim=1)
        fused, modality_scores, entropy, confidence = self.fusion(
            audio_repr,
            video_repr,
            text_repr,
            modality_present=modality_present,
        )

        return {
            "phq_total": self.phq_head(fused).squeeze(-1),
            "phq_questions": self.question_head(fused),
            "classification": self.classification_head(fused).squeeze(-1),
            "confidence": self.confidence_head(fused).squeeze(-1),
            "modality_scores": modality_scores,
            "modality_confidence": confidence,
            "fusion_entropy": entropy,
            "entropy": entropy,
        }
