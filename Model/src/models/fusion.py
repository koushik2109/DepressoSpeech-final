from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for multimodal fusion.
    
    Allows each modality to attend to other modalities for better integration.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        modality_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply cross-modal attention.
        
        Args:
            modality_tokens: [batch_size, num_modalities, embed_dim]
            key_padding_mask: [batch_size, num_modalities] boolean mask
            
        Returns:
            Attended tokens [batch_size, num_modalities, embed_dim]
        """
        attended, _ = self.attention(
            modality_tokens,
            modality_tokens,
            modality_tokens,
            key_padding_mask=key_padding_mask,
        )
        return self.proj(attended + modality_tokens)


class ModalityGating(nn.Module):
    """Learnable modality gating without fixed floor constraints.
    
    Learns to weight each modality dynamically while respecting presence masks.
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.entropy_weight = entropy_weight
        
        # Per-modality gating networks
        self.modality_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(3)
            ]
        )

    def forward(
        self,
        modality_reprs: list[torch.Tensor],
        modality_present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gating weights for modalities.
        
        Args:
            modality_reprs: List of [batch_size, embed_dim] tensors for each modality
            modality_present: [batch_size, 3] boolean mask of which modalities are present
            
        Returns:
            Tuple of:
                - weights: [batch_size, 3] normalized weights
                - entropy: scalar entropy regularization loss
        """
        # Compute logits for each modality
        logits = torch.cat(
            [head(mod) for head, mod in zip(self.modality_heads, modality_reprs)],
            dim=-1,  # [batch_size, 3]
        )

        # Apply presence masking before softmax
        if modality_present is not None:
            # Mask out absent modalities
            logits = logits.masked_fill(~modality_present, float("-1e9"))

        # Compute weights via softmax
        weights = F.softmax(logits, dim=-1)  # [batch_size, 3]

        # Soft gate floor: ensure every *present* modality gets at least min_w weight.
        # This prevents modality collapse (e.g. audio gate → 0.99) during training.
        # Text is encouraged toward higher values via the text_gate_weight loss term
        # in the loss function — not through a hard architectural floor, which would
        # limit the residual learning budget and reduce overall CCC.
        min_w = 0.15
        if modality_present is not None:
            present_f = modality_present.float()  # [B, 3]
            num_present = present_f.sum(dim=-1, keepdim=True).clamp(min=1.0)

            # Floor: present modalities each get at least min_w, absent ones get 0.
            # Renormalize so weights still sum to 1.
            floored = weights * present_f  # zero out absent modalities
            floor_values = min_w * present_f  # minimum per present modality
            # Total floor mass consumed by floor
            floor_sum = floor_values.sum(dim=-1, keepdim=True)  # min_w * num_present
            # Remaining mass to distribute via learned weights above the floor
            remaining = (1.0 - floor_sum).clamp(min=0.0)
            # Residual learned weights (above floor), re-normalised
            residual_weights = (floored - floor_values).clamp(min=0.0)
            residual_sum = residual_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            residual_norm = residual_weights / residual_sum * remaining
            weights = floor_values + residual_norm

            # Fallback: if all modalities absent, use uniform
            no_present_rows = (present_f.sum(dim=-1) < 1e-6)
            if no_present_rows.any():
                fallback = torch.ones_like(weights) / 3.0
                weights[no_present_rows] = fallback[no_present_rows]
        else:
            # No presence mask — apply floor uniformly across all 3
            floor_values = torch.full_like(weights, min_w)
            remaining = 1.0 - min_w * 3
            residual = (weights - floor_values).clamp(min=0.0)
            residual_sum = residual.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            residual_norm = residual / residual_sum * remaining
            weights = floor_values + residual_norm

        # Compute entropy (max = log(3)≈1.099 for uniform distribution)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()

        return weights, entropy


class ModalityConfidence(nn.Module):
    """Per-modality confidence estimation for uncertainty awareness."""
    
    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.audio_confidence = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.video_confidence = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.text_confidence = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        audio_repr: torch.Tensor,
        video_repr: torch.Tensor,
        text_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute confidence scores for each modality.
        
        Returns:
            Confidence scores [batch_size, 3]
        """
        audio_conf = self.audio_confidence(audio_repr).squeeze(-1)  # [batch_size]
        video_conf = self.video_confidence(video_repr).squeeze(-1)
        text_conf = self.text_confidence(text_repr).squeeze(-1)
        
        return torch.stack([audio_conf, video_conf, text_conf], dim=-1)  # [batch_size, 3]


class MultimodalFusion(nn.Module):
    """Enhanced multimodal fusion with entropy regularization and confidence scores.
    
    Supports multiple fusion strategies:
    - early: Concatenate all modalities before fusion
    - late: Gate modalities then concatenate with individual representations
    - hybrid: Use cross-modal attention then gating (default, best for small datasets)
    """

    def __init__(
        self,
        embed_dim: int,
        fusion_dim: int,
        mode: str = "hybrid",
        dropout: float = 0.2,
        num_attention_heads: int = 4,
        entropy_weight: float = 0.1,
    ):
        """Initialize multimodal fusion.
        
        Args:
            embed_dim: Individual modality embedding dimension
            fusion_dim: Fusion output dimension
            mode: Fusion strategy ('early', 'late', 'hybrid')
            dropout: Dropout rate
            num_attention_heads: Number of attention heads
            entropy_weight: Weight for entropy regularization (encourages diversity)
        """
        super().__init__()
        self.mode = mode
        self.entropy_weight = entropy_weight

        # Gating mechanism (no min_gate floor - entropy regularization handles balance)
        self.gating = ModalityGating(
            feature_dim=embed_dim,
            hidden_dim=fusion_dim,
            dropout=dropout,
            entropy_weight=entropy_weight,
        )

        # Cross-modal attention (optional, helps with hybrid mode)
        self.cross_attention = CrossModalAttention(
            embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Confidence scoring
        self.confidence = ModalityConfidence(embed_dim, hidden_dim=fusion_dim, dropout=dropout)

        # Determine fusion projection input dimension
        if mode == "early":
            projection_input_dim = embed_dim * 3
        else:  # late or hybrid
            projection_input_dim = embed_dim * 4  # gated_fusion + 3 individual reps

        # Fusion projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(projection_input_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim),
        )

        # Final fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim),
        )

    def forward(
        self,
        audio_repr: torch.Tensor,
        video_repr: torch.Tensor,
        text_repr: torch.Tensor,
        modality_present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Perform multimodal fusion.
        
        Args:
            audio_repr: Audio representation [batch_size, embed_dim]
            video_repr: Video representation [batch_size, embed_dim]
            text_repr: Text representation [batch_size, embed_dim]
            modality_present: [batch_size, 3] boolean mask of present modalities
            
        Returns:
            Tuple of:
                - fused: Fused representation [batch_size, fusion_dim]
                - modality_scores: Dict with per-modality scores
                - entropy: Entropy regularization loss
                - confidence: Per-modality confidence [batch_size, 3]
        """
        modality_reprs = [audio_repr, video_repr, text_repr]

        # Compute gating weights with entropy regularization
        weights, entropy = self.gating(modality_reprs, modality_present)

        # Compute per-modality confidence scores
        confidence = self.confidence(audio_repr, video_repr, text_repr)

        # Apply confidence-aware gating (emphasize confident modalities)
        weighted_confidence = confidence * weights
        confidence_sum = weighted_confidence.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        confidence_aware_weights = weighted_confidence / confidence_sum

        modality_scores = {
            "audio": weights[:, 0].detach(),
            "video": weights[:, 1].detach(),
            "text": weights[:, 2].detach(),
            "audio_confidence": confidence[:, 0].detach(),
            "video_confidence": confidence[:, 1].detach(),
            "text_confidence": confidence[:, 2].detach(),
            # Non-detached weights so gate_balance_loss can backprop
            "raw_weights": weights,
        }

        # Apply fusion strategy
        if self.mode == "early":
            # Early fusion: just concatenate
            fused_input = torch.cat(modality_reprs, dim=-1)
            fused = self.fusion_projection(fused_input)

        elif self.mode == "late":
            # Late fusion: gate then concatenate
            gated = (
                audio_repr * weights[:, 0:1]
                + video_repr * weights[:, 1:2]
                + text_repr * weights[:, 2:3]
            )
            fused_input = torch.cat([gated, audio_repr, video_repr, text_repr], dim=-1)
            fused = self.fusion_projection(fused_input)

        else:  # hybrid (default)
            # Hybrid: cross-modal attention + gating
            # Stack modalities for attention
            tokens = torch.stack(modality_reprs, dim=1)  # [batch_size, 3, embed_dim]
            
            # Create attention mask if modalities are missing
            key_padding_mask = None
            if modality_present is not None:
                key_padding_mask = ~modality_present  # [batch_size, 3]

            # Apply cross-modal attention
            attended = self.cross_attention(tokens, key_padding_mask=key_padding_mask)

            # Apply gating weights to attended tokens
            gated = attended * weights.unsqueeze(-1)
            gated_fusion = gated.sum(dim=1)  # [batch_size, embed_dim]

            # Concatenate gated fusion with individual representations
            fused_input = torch.cat([gated_fusion, audio_repr, video_repr, text_repr], dim=-1)
            fused = self.fusion_projection(fused_input)

        # Apply final fusion head
        fused = self.fusion_head(fused)

        return fused, modality_scores, entropy, confidence