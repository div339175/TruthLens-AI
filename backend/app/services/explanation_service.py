"""Explanation service formatting and validating model interpretability artifacts."""

from typing import Dict, Any, List
from backend.app.schemas.verification import ModalityEvidence, TokenImportance


class ExplanationService:
    def format_text_explanation(self, tokens: List[Dict[str, Any]]) -> ModalityEvidence:
        token_imps = [TokenImportance(token=t["token"], score=t["score"]) for t in tokens]
        return ModalityEvidence(
            modality="text",
            type="model-derived",
            description="Token attribution via embedding gradients highlights suspicious textual segments.",
            token_importance=token_imps,
        )

    def format_visual_explanation(self, cam_uri: str, meta: Dict[str, Any]) -> ModalityEvidence:
        return ModalityEvidence(
            modality="visual",
            type="model-derived",
            description="Grad-CAM highlights spatially concentrated visual activation in final convolutional layer.",
            heatmap_available=True,
            heatmap_base64=cam_uri,
            details=meta,
        )
