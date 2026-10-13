"""Text Explainability: Token attribution and saliency for deepfake and misinformation analysis."""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from ml.preprocessing.text import TextPreprocessor


class TextExplainer:
    """Computes token attribution using embedding-level gradient saliency."""

    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor

    def explain_instance(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int = 1,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Compute saliency scores for individual text tokens using gradient magnitude.

        Args:
            model: MultimodalVerificationModel or unimodal text model
            input_ids: [1, seq_len]
            attention_mask: [1, seq_len]
            target_class: Class index to explain (default 1: FAKE)
            top_k: Number of highest scoring tokens to return

        Returns:
            List of dicts: [{"token": str, "score": float}, ...]
        """
        model.eval()

        # Extract embeddings through the transformer embedding layer with grad enabled
        embeddings_layer = model.text_encoder.transformer.get_input_embeddings()
        token_embeddings = embeddings_layer(input_ids).clone().detach().requires_grad_(True)

        # Forward pass through remainder of transformer
        # In HuggingFace, models accept inputs_embeds
        transformer_out = model.text_encoder.transformer(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
        )
        seq_output = transformer_out.last_hidden_state
        pooled = seq_output[:, 0, :]
        proj = model.text_encoder.projection(pooled)

        # Check whether unimodal or multimodal
        if hasattr(model, "text_classifier"):
            logits = model.text_classifier(proj)
        elif hasattr(model, "classifier"):
            logits = model.classifier(proj)
        else:
            return []

        target_score = logits[0, target_class]
        target_score.backward()

        if token_embeddings.grad is None:
            # Fallback if grad unavailable
            return self._heuristic_fallback(input_ids, top_k)

        # Saliency: L2 norm of gradient across embedding dimension
        grads = token_embeddings.grad[0]  # [seq_len, embed_dim]
        saliency = torch.norm(grads, dim=-1)  # [seq_len]

        # Mask out padding and special tokens
        tokens = self.preprocessor.decode_tokens(input_ids[0])
        token_scores: List[Tuple[str, float]] = []

        max_val = torch.max(saliency).item() if torch.max(saliency).item() > 0 else 1.0

        for idx, (token, mask_val) in enumerate(zip(tokens, attention_mask[0].tolist())):
            if mask_val == 0 or token in ["[PAD]", "[CLS]", "[SEP]"]:
                continue
            normalized_score = round(float(saliency[idx].item() / max_val), 4)
            token_scores.append((token, normalized_score))

        # Sort descending by attribution score
        token_scores.sort(key=lambda x: x[1], reverse=True)
        top_tokens = token_scores[:top_k]

        return [{"token": t, "score": s} for t, s in top_tokens]

    def _heuristic_fallback(self, input_ids: torch.Tensor, top_k: int) -> List[Dict[str, Any]]:
        """Transparent heuristic fallback if gradient calculation is disabled."""
        tokens = self.preprocessor.decode_tokens(input_ids[0])
        valid = [t for t in tokens if t not in ["[PAD]", "[CLS]", "[SEP]"]]
        return [{"token": t, "score": 0.5} for t in valid[:top_k]]
