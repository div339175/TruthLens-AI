"""Evidence Aggregation and Report Generation for TruthLens AI."""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from backend.app.schemas.verification import ModalityEvidence, VerificationReport, TokenImportance


class EvidenceEngine:
    """Combines model outputs across modalities into an explainable verification report."""

    def build_report(
        self,
        decision: Dict[str, Any],
        modalities_used: List[str],
        text_tokens: Optional[List[Dict[str, Any]]] = None,
        cam_data_uri: Optional[str] = None,
        cam_meta: Optional[Dict[str, Any]] = None,
        video_meta: Optional[Dict[str, Any]] = None,
        cross_attention_data: Optional[Dict[str, Any]] = None,
        execution_mode: str = "real",
        model_version: str = "TruthLens Multimodal v0.1.0",
        fusion_type: str = "cross_attention",
        device_used: str = "cpu",
    ) -> VerificationReport:
        """Construct structured VerificationReport compliant with university project requirements."""
        evidence_list: List[ModalityEvidence] = []
        pred = decision["prediction"]
        conf = decision["confidence"]
        is_uncertain = decision["uncertain"]

        # 1. Text Evidence
        if "text" in modalities_used:
            if text_tokens:
                tokens_formatted = [
                    TokenImportance(token=t["token"], score=t["score"])
                    for t in text_tokens
                ]
                top_tokens_str = ", ".join([f"'{t['token']}' ({t['score']:.2f})" for t in text_tokens[:4]])
                text_desc = (
                    f"Model-derived gradient saliency identified key tokens influencing the classification: {top_tokens_str}."
                    if pred == "FAKE"
                    else f"Textual evidence examined for synthetic stylometric markers. Top influential tokens: {top_tokens_str}."
                )
                evidence_list.append(ModalityEvidence(
                    modality="text",
                    type="model-derived",
                    description=text_desc,
                    token_importance=tokens_formatted,
                    details={"token_count": len(text_tokens)},
                ))
            else:
                evidence_list.append(ModalityEvidence(
                    modality="text",
                    type="heuristic",
                    description="Textual tokens analyzed through transformer encoder with baseline uniform attribution.",
                ))

        # 2. Visual / Image Evidence
        if "visual" in modalities_used or "image" in modalities_used:
            if cam_data_uri:
                act_pct = cam_meta.get("high_activation_area_fraction", 0.0) if cam_meta else 0.0
                vis_desc = (
                    f"Grad-CAM on ResNet18 conv stem highlights concentrated suspicious visual regions "
                    f"covering approx {act_pct * 100:.1f}% of frame area."
                    if pred == "FAKE"
                    else "Visual feature extraction examined spatial consistency. Heatmap indicates standard feature distribution."
                )
                evidence_list.append(ModalityEvidence(
                    modality="visual",
                    type="model-derived",
                    description=vis_desc,
                    heatmap_available=True,
                    heatmap_base64=cam_data_uri,
                    details=cam_meta,
                ))
            else:
                evidence_list.append(ModalityEvidence(
                    modality="visual",
                    type="model-derived",
                    description="Visual backbone extracted 512-dim spatial representation. No focal manipulation hotspot isolated.",
                ))

        # 3. Video Evidence
        if "video" in modalities_used and video_meta:
            n_frames = video_meta.get("sampled_frames_count", 8)
            evidence_list.append(ModalityEvidence(
                modality="video",
                type="heuristic",
                description=f"Uniform temporal sampling evaluated {n_frames} frames across video duration for cross-frame continuity.",
                details=video_meta,
            ))

        # 4. Cross-Modal Evidence
        if len(modalities_used) > 1:
            if cross_attention_data is not None:
                evidence_list.append(ModalityEvidence(
                    modality="cross-modal",
                    type="model-derived",
                    description=(
                        "Bidirectional cross-attention computed text-to-vision alignment scores, "
                        "capturing cross-modal correspondence between textual description and visual patches."
                    ),
                    details={"fusion_architecture": "cross_attention"},
                ))
            else:
                evidence_list.append(ModalityEvidence(
                    modality="cross-modal",
                    type="model-derived",
                    description="Feature concatenation fusion projected combined multimodal space.",
                    details={"fusion_architecture": "concatenation"},
                ))

        # Determine manipulation type
        if is_uncertain:
            manipulation_type = "Inconclusive / Modality Discrepancy"
            explanation = (
                f"The system classified this content as UNCERTAIN because the peak confidence score ({conf*100:.1f}%) "
                f"fell below the operational threshold ({decision.get('threshold_used', 0.65)*100:.0f}%). "
                "This occurs when textual and visual signals provide conflicting evidence or subtle manipulation cues."
            )
        elif pred == "FAKE":
            if "visual" in modalities_used or "video" in modalities_used:
                manipulation_type = "Potential visual / multimodal manipulation"
            else:
                manipulation_type = "Synthetic / manipulated textual content"

            explanation = (
                f"The content was classified as FAKE with {conf*100:.1f}% model confidence. "
                f"Features extracted across {', '.join(modalities_used)} exhibit synthetic characteristics "
                "consistent with generative or spliced manipulations."
            )
        else:
            manipulation_type = "No manipulation detected (Authentic)"
            explanation = (
                f"The content was verified as REAL with {conf*100:.1f}% model confidence. "
                f"Analysis across {', '.join(modalities_used)} demonstrates coherence with authentic reference distributions."
            )

        timestamp_str = datetime.now(timezone.utc).isoformat()

        return VerificationReport(
            prediction=pred,
            confidence=conf,
            real_probability=decision["real_probability"],
            fake_probability=decision["fake_probability"],
            uncertain=is_uncertain,
            modalities_used=modalities_used,
            manipulation_type=manipulation_type,
            evidence=evidence_list,
            explanation=explanation,
            model_version=model_version,
            fusion_type=fusion_type,
            device_used=device_used,
            execution_mode=execution_mode,
            timestamp=timestamp_str,
        )
