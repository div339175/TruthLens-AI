"""Text Encoder module using Hugging Face Transformers."""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TextEncoder(nn.Module):
    """Transformer-based text encoder returning pooled representations and sequence features."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        projection_dim: int = 256,
        fine_tune: bool = False,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            attn_implementation="eager",
        )
        self.embed_dim = self.config.hidden_size

        # Freeze transformer if not fine-tuning
        if not fine_tune:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Projection layer to shared multimodal space
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            output_hidden_states: whether to return hidden states for saliency / interpretability

        Returns:
            Dict containing:
                - 'sequence_output': [B, seq_len, embed_dim]
                - 'pooled_output': [B, embed_dim] (CLS token or mean pooled)
                - 'projected_pooled': [B, projection_dim]
                - 'projected_sequence': [B, seq_len, projection_dim]
        """
        transformer_out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        seq_output = transformer_out.last_hidden_state  # [B, seq_len, D]

        # Pooled representation (DistilBERT uses first token [CLS])
        pooled = seq_output[:, 0, :]  # [B, D]

        proj_pooled = self.projection(pooled)
        proj_seq = self.projection(seq_output)

        result: Dict[str, torch.Tensor] = {
            "sequence_output": seq_output,
            "pooled_output": pooled,
            "projected_pooled": proj_pooled,
            "projected_sequence": proj_seq,
        }

        if output_hidden_states and hasattr(transformer_out, "hidden_states"):
            result["hidden_states"] = transformer_out.hidden_states

        return result
