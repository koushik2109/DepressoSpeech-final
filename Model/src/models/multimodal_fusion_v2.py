"""
Trimodal Fusion Model for Depression Severity Prediction (V2).

Architecture:
    Audio:  (B,N,208) → AudioEncoder → (B,64)  → proj(64,D)  → (B,D)
    Video:  (B,T,49+512) → VideoEncoder → (B,256) → proj(256,D)  → (B,D)
    Text:   (B,N,384) → StatsPool(mean+std=768) → proj(768,D) → (B,D)

    Fusion: Cross-Modal Attention + Learned Modality Gates
            [audio_repr, video_repr, text_repr] → MultiheadAttention
            → Gated fusion → (B, D) → Linear → PHQ-8 score

Key Features:
    1. Attention-based fusion: learns cross-modal interactions
    2. Missing modality handling: zero-masks absent modalities with learned tokens
    3. Modality contribution tracking: outputs per-modality contribution scores
    4. Backward compatible: can run audio+text only (degrades to V1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.models.statistics_pooling import StatisticsPooling
from src.models.multimodal_fusion import AudioEncoder
from src.models.video_encoder import VideoEncoder

logger = logging.getLogger(__name__)


@dataclass
class FusionOutput:
    """Structured output from the trimodal fusion model."""
    prediction: torch.Tensor        # (B, 1) — PHQ-8 score
    audio_contribution: float       # 0-1, relative audio influence
    video_contribution: float       # 0-1, relative video influence
    text_contribution: float        # 0-1, relative text influence
    gate_values: Dict[str, float]   # raw gate values for debugging


class CrossModalAttention(nn.Module):
    """
    Multi-head cross-modal attention for fusing modality representations.

    Input: List of modality embeddings, each (B, D)
    Output: Fused representation (B, D)

    Uses self-attention across modalities (not time) — each modality
    attends to all others to learn cross-modal interactions.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        modality_embeddings: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_embeddings: (B, num_modalities, D)
            key_padding_mask: (B, num_modalities) — True for absent modalities

        Returns:
            fused: (B, num_modalities, D) — attention-fused representations
            attn_weights: (B, num_modalities, num_modalities)
        """
        # Self-attention across modalities
        attended, attn_weights = self.attention(
            modality_embeddings, modality_embeddings, modality_embeddings,
            key_padding_mask=key_padding_mask,
        )
        # Residual + norm
        x = self.norm(modality_embeddings + attended)
        # FFN + residual
        x = self.ffn_norm(x + self.ffn(x))
        return x, attn_weights


class TrimodalFusionModel(nn.Module):
    """
    Production-ready trimodal fusion for depression severity prediction.

    Supports Audio + Video + Text with graceful degradation when modalities
    are missing (uses learned modality tokens as replacements).

    Input shapes:
        audio:    mfcc (B,N,120) + egemaps (B,N,88) + behavioral (B,16)
        video:    openface (B,T,49) + cnn_embed (B,T,512)
        text:     embeddings (B,N,384)

    Output:
        FusionOutput with PHQ-8 prediction and modality contributions
    """

    # Feature dimensions
    MFCC_DIM = 120
    EGEMAPS_DIM = 88
    AUDIO_DIM = MFCC_DIM + EGEMAPS_DIM  # 208
    TEXT_DIM = 384
    OPENFACE_DIM = 49
    CNN_DIM = 512
    BEHAVIORAL_DIM = 16

    def __init__(
        self,
        fusion_dim: int = 128,
        num_attention_heads: int = 4,
        stats_mode: str = "mean_std",
        dropout: float = 0.1,
        modality_dropout: float = 0.15,
        openface_dim: int = 49,
        cnn_dim: int = 512,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.modality_dropout = modality_dropout

        # ═══ Audio Branch ═══
        self.audio_encoder = AudioEncoder(
            input_dim=self.AUDIO_DIM, hidden_dim=64, out_dim=32,
        )
        audio_out = self.audio_encoder.output_dim + self.BEHAVIORAL_DIM  # 64+16=80
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_out, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ═══ Video Branch ═══
        self.video_encoder = VideoEncoder(
            openface_dim=openface_dim,
            cnn_dim=cnn_dim,
            proj_dim=64,
            out_dim=128,
            stats_mode=stats_mode,
            dropout=dropout,
        )
        self.video_proj = nn.Sequential(
            nn.Linear(self.video_encoder.output_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ═══ Text Branch ═══
        self.text_pool = StatisticsPooling(self.TEXT_DIM, stats=stats_mode)
        text_pooled_dim = self.text_pool.output_dim  # 768
        self.text_proj = nn.Sequential(
            nn.BatchNorm1d(text_pooled_dim),
            nn.Linear(text_pooled_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ═══ Learned Modality Tokens (for missing modality replacement) ═══
        self.audio_token = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        self.video_token = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        self.text_token = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)

        # ═══ Modality Presence Embeddings ═══
        # Indicate which modality is present vs replaced by token
        self.modality_type_embed = nn.Embedding(3, fusion_dim)  # 0=audio, 1=video, 2=text

        # ═══ Cross-Modal Attention Fusion ═══
        self.cross_attention = CrossModalAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # ═══ Modality Gates (learned importance weighting) ═══
        self.modality_gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 3),
            nn.Softmax(dim=-1),
        )

        # ═══ Prediction Head ═══
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

        # ═══ Modality-specific fallback heads (for single-modality inference) ═══
        self.audio_head = nn.Linear(fusion_dim, 1)
        self.video_head = nn.Linear(fusion_dim, 1)
        self.text_head = nn.Linear(fusion_dim, 1)

        self._init_weights()

        # Initialize near-zero for smoother training start
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"TrimodalFusionModel: fusion_dim={fusion_dim}, heads={num_attention_heads}, "
            f"modality_dropout={modality_dropout}, total_params={total_params:,}"
        )

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2 and "attention" not in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name and "attention" not in name:
                nn.init.zeros_(param)

    def _encode_audio(
        self,
        mfcc: torch.Tensor,
        egemaps: torch.Tensor,
        mask: torch.Tensor,
        behavioral: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio features → (B, fusion_dim)."""
        audio = torch.cat([mfcc, egemaps], dim=-1)  # (B, N, 208)
        audio_repr = self.audio_encoder(audio, mask)  # (B, 64)

        if behavioral is not None:
            audio_repr = torch.cat([audio_repr, behavioral], dim=-1)  # (B, 80)
        else:
            audio_repr = torch.cat(
                [audio_repr, torch.zeros(audio_repr.shape[0], self.BEHAVIORAL_DIM, device=audio_repr.device)],
                dim=-1,
            )

        return self.audio_proj(audio_repr)  # (B, fusion_dim)

    def _encode_video(
        self,
        openface: torch.Tensor,
        cnn_embed: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode video features → (B, fusion_dim)."""
        video_repr = self.video_encoder(openface, cnn_embed, mask)
        return self.video_proj(video_repr)  # (B, fusion_dim)

    def _encode_text(
        self,
        text: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text features → (B, fusion_dim)."""
        text_pooled = self.text_pool(text, mask)  # (B, 768)
        return self.text_proj(text_pooled)  # (B, fusion_dim)

    def forward(
        self,
        # Audio inputs
        mfcc: Optional[torch.Tensor] = None,
        egemaps: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        behavioral: Optional[torch.Tensor] = None,
        # Video inputs
        openface: Optional[torch.Tensor] = None,
        cnn_embed: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        # Text inputs
        text: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> FusionOutput:
        """
        Forward pass supporting any combination of modalities.

        At least one modality must be provided. Missing modalities are
        replaced by learned tokens and masked in the attention layer.

        Returns:
            FusionOutput with prediction and modality contributions
        """
        B = self._infer_batch_size(mfcc, openface, text)
        device = self._infer_device(mfcc, openface, text)

        # ─── Determine which modalities are present ───
        has_audio = (mfcc is not None and egemaps is not None)
        has_video = (openface is not None and cnn_embed is not None)
        has_text = (text is not None)

        if not (has_audio or has_video or has_text):
            raise ValueError("At least one modality must be provided")

        # ─── Encode each modality or use learned token ───
        if has_audio:
            audio_repr = self._encode_audio(mfcc, egemaps, audio_mask, behavioral)
        else:
            audio_repr = self.audio_token.expand(B, -1, -1).squeeze(1)  # (B, D)

        if has_video:
            video_repr = self._encode_video(openface, cnn_embed, video_mask)
        else:
            video_repr = self.video_token.expand(B, -1, -1).squeeze(1)

        if has_text:
            text_repr = self._encode_text(text, text_mask)
        else:
            text_repr = self.text_token.expand(B, -1, -1).squeeze(1)

        # ─── Modality dropout (training only) ───
        if self.training and self.modality_dropout > 0:
            audio_repr, video_repr, text_repr, has_audio, has_video, has_text = (
                self._apply_modality_dropout(
                    audio_repr, video_repr, text_repr,
                    has_audio, has_video, has_text, B, device,
                )
            )

        # ─── Add modality type embeddings ───
        type_ids = torch.arange(3, device=device)
        type_embeds = self.modality_type_embed(type_ids)  # (3, D)

        audio_repr = audio_repr + type_embeds[0]
        video_repr = video_repr + type_embeds[1]
        text_repr = text_repr + type_embeds[2]

        # ─── Stack into sequence for cross-modal attention ───
        # (B, 3, D) — three modality tokens
        modality_seq = torch.stack([audio_repr, video_repr, text_repr], dim=1)

        # Build padding mask: True = ignore (absent modality)
        padding_mask = torch.tensor(
            [[not has_audio, not has_video, not has_text]] * B,
            dtype=torch.bool,
            device=device,
        )
        # If all modalities are masked (shouldn't happen), unmask all
        if padding_mask.all():
            padding_mask = torch.zeros(B, 3, dtype=torch.bool, device=device)

        # ─── Cross-Modal Attention ───
        attended, attn_weights = self.cross_attention(modality_seq, padding_mask)
        # attended: (B, 3, D), attn_weights: (B, 3, 3)

        # ─── Modality Gate (learned importance) ───
        gate_input = attended.reshape(B, -1)  # (B, 3*D)
        gate_weights = self.modality_gate(gate_input)  # (B, 3) — softmax

        # Weighted combination
        fused = (
            gate_weights[:, 0:1] * attended[:, 0] +
            gate_weights[:, 1:2] * attended[:, 1] +
            gate_weights[:, 2:3] * attended[:, 2]
        )  # (B, D)

        # ─── Prediction ───
        prediction = self.head(fused)  # (B, 1)

        # ─── Compute modality contributions ───
        gate_mean = gate_weights.mean(dim=0)  # (3,)
        audio_contrib = float(gate_mean[0].item()) if has_audio else 0.0
        video_contrib = float(gate_mean[1].item()) if has_video else 0.0
        text_contrib = float(gate_mean[2].item()) if has_text else 0.0

        # Renormalize contributions to sum to 1
        total = audio_contrib + video_contrib + text_contrib
        if total > 0:
            audio_contrib /= total
            video_contrib /= total
            text_contrib /= total

        return FusionOutput(
            prediction=prediction,
            audio_contribution=round(audio_contrib, 4),
            video_contribution=round(video_contrib, 4),
            text_contribution=round(text_contrib, 4),
            gate_values={
                "audio_gate": round(float(gate_mean[0].item()), 4),
                "video_gate": round(float(gate_mean[1].item()), 4),
                "text_gate": round(float(gate_mean[2].item()), 4),
                "attention_weights": attn_weights.detach().cpu().tolist() if attn_weights is not None else None,
            },
        )

    def predict_single_modality(
        self,
        modality: str,
        features: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Predict using only one modality (fallback mode).

        Args:
            modality: "audio", "video", or "text"
            features: Pre-processed features
            mask: Validity mask
        """
        if modality == "audio":
            mfcc = kwargs.get("mfcc", features[:, :, :self.MFCC_DIM])
            egemaps = kwargs.get("egemaps", features[:, :, self.MFCC_DIM:])
            repr_ = self._encode_audio(mfcc, egemaps, mask, kwargs.get("behavioral"))
            return self.audio_head(repr_)
        elif modality == "video":
            openface = kwargs.get("openface", features[:, :, :self.OPENFACE_DIM])
            cnn_embed = kwargs.get("cnn_embed", features[:, :, self.OPENFACE_DIM:])
            repr_ = self._encode_video(openface, cnn_embed, mask)
            return self.video_head(repr_)
        elif modality == "text":
            repr_ = self._encode_text(features, mask)
            return self.text_head(repr_)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _apply_modality_dropout(
        self, audio_repr, video_repr, text_repr,
        has_audio, has_video, has_text, B, device,
    ):
        """Randomly drop modalities during training for robustness."""
        present_count = sum([has_audio, has_video, has_text])
        if present_count <= 1:
            return audio_repr, video_repr, text_repr, has_audio, has_video, has_text

        drop_rand = torch.rand(3, device=device)

        # Only drop if at least 2 modalities will remain
        if has_audio and drop_rand[0] < self.modality_dropout and present_count > 2:
            audio_repr = self.audio_token.expand(B, -1, -1).squeeze(1)
            has_audio = False
            present_count -= 1

        if has_video and drop_rand[1] < self.modality_dropout and present_count > 1:
            video_repr = self.video_token.expand(B, -1, -1).squeeze(1)
            has_video = False
            present_count -= 1

        if has_text and drop_rand[2] < self.modality_dropout * 0.5 and present_count > 1:
            text_repr = self.text_token.expand(B, -1, -1).squeeze(1)
            has_text = False

        return audio_repr, video_repr, text_repr, has_audio, has_video, has_text

    @staticmethod
    def _infer_batch_size(*tensors) -> int:
        for t in tensors:
            if t is not None:
                return t.shape[0]
        return 1

    @staticmethod
    def _infer_device(*tensors) -> torch.device:
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")

    def param_summary(self) -> Dict[str, int]:
        def _count(module, trainable_only=False):
            return sum(
                p.numel() for p in module.parameters()
                if not trainable_only or p.requires_grad
            )

        return {
            "audio_encoder": _count(self.audio_encoder),
            "audio_proj": _count(self.audio_proj),
            "video_encoder": _count(self.video_encoder),
            "video_proj": _count(self.video_proj),
            "text_pool": 0,
            "text_proj": _count(self.text_proj),
            "cross_attention": _count(self.cross_attention),
            "modality_gate": _count(self.modality_gate),
            "head": _count(self.head),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
