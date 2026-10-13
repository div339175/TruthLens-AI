"""Bidirectional Cross-Attention module for multimodal alignment and fusion."""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """Single directional cross-attention layer: Query attends to Key/Value."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

        query: [B, seq_len_q, D]
        key_value: [B, seq_len_kv, D]
        key_padding_mask: [B, seq_len_kv] (True for padded positions)

        Returns:
            output: [B, seq_len_q, D]
            attn_weights: [B, seq_len_q, seq_len_kv]
        """
        # Cross-attention residual
        attn_out, attn_weights = self.mha(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = self.norm1(query + self.dropout(attn_out))

        # FFN residual
        ffn_out = self.ffn(x)
        out = self.norm2(x + self.dropout(ffn_out))
        return out, attn_weights


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional Cross-Attention between Text sequence and Visual spatial tokens.

    - Text attends to Visual patches (Text -> Visual)
    - Visual patches attend to Text tokens (Visual -> Text)
    - Both attended sequences are pooled and fused.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.text_to_vision = CrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.vision_to_text = CrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Fusion projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_seq: torch.Tensor,
        vision_seq: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Args:

        text_seq: [B, text_len, embed_dim]
        vision_seq: [B, num_patches, embed_dim]
        text_padding_mask: [B, text_len] bool (True where mask==0)

        Returns:
            Dict containing:
                - 'fused_embedding': [B, embed_dim]
                - 'text_attended': [B, text_len, embed_dim]
                - 'vision_attended': [B, num_patches, embed_dim]
                - 'text_to_vision_attn': [B, text_len, num_patches]
                - 'vision_to_text_attn': [B, num_patches, text_len]
        """
        # Text queries attending to Vision
        text_att, t2v_weights = self.text_to_vision(query=text_seq, key_value=vision_seq)

        # Vision queries attending to Text
        v2t_key_mask = text_padding_mask if text_padding_mask is not None else None
        vision_att, v2t_weights = self.vision_to_text(
            query=vision_seq,
            key_value=text_seq,
            key_padding_mask=v2t_key_mask,
        )

        # Pool attended representations across token/patch length
        text_pooled = torch.mean(text_att, dim=1)  # [B, embed_dim]
        vision_pooled = torch.mean(vision_att, dim=1)  # [B, embed_dim]

        # Final cross-modal representation
        fused = self.fusion_projection(torch.cat([text_pooled, vision_pooled], dim=-1))

        return {
            "fused_embedding": fused,
            "text_attended": text_att,
            "vision_attended": vision_att,
            "text_to_vision_attn": t2v_weights,
            "vision_to_text_attn": v2t_weights,
        }
