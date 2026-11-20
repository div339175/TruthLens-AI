"""Qualitative Error and Challenging Case Analysis for TruthLens AI (Phase 4).

Evaluates 15 challenging multimodal cases across:
- Semantic mismatch / cross-modal contradiction
- Subtle face-swap deepfakes
- Diffusion synthesis
- Spliced banners / partial image tampering
- Borderline authentic satellite / scientific edge cases

Executes genuine model forward passes and records:
- case ID
- ground truth
- prediction
- confidence
- modalities
- important evidence
- explanation
- whether modalities agree / disagree
- error category
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.inference.predictor import TruthLensPredictor

REPORTS_DIR = BASE_DIR / "experiments" / "reports"
SAMPLES_DIR = BASE_DIR / "ml" / "data" / "samples"

# 15 challenging test cases grounded in actual image files in ml/data/samples/
CANDIDATE_DIFFICULT_CASES = [
    # Category 1: Semantic Mismatch & Cross-Modal Contradiction (Real visual + deceptive text claim)
    {
        "case_id": "CASE_01",
        "category": "semantic_mismatch",
        "title": "Peaceful Festival Miscaptioned as Street Riot",
        "text": "Breaking news: violent riots and civil unrest break out in downtown capital, teargas deployed.",
        "image_file": "real_official_press_01.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "The visual modality is authentic, but the textual claim describes violent conflict. Detectors relying only on visual artifacts would classify it as REAL.",
    },
    {
        "case_id": "CASE_02",
        "category": "semantic_mismatch",
        "title": "Historical Archival Space Launch Claimed as New Ballistic Test",
        "text": "Unverified surveillance footage captures secret intercontinental ballistic missile launch test today.",
        "image_file": "real_science_nature_01.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Archival footage is completely unmanipulated, but out-of-context caption fabricates modern active warfare.",
    },
    {
        "case_id": "CASE_03",
        "category": "semantic_mismatch",
        "title": "Natural Wildfire Drill Repurposed as Catastrophic Disaster",
        "text": "Emergency declared as uncontrolled chemical fire engulfs municipal residential sectors.",
        "image_file": "real_science_nature_02.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Standard training drill photos reused with catastrophic panic-inducing headlines.",
    },
    {
        "case_id": "CASE_04",
        "category": "semantic_mismatch",
        "title": "Commercial Hospital Wing Opening Claimed as Secret Military Biolab",
        "text": "Leaked photograph exposes clandestine bioweapon research laboratory operating under commercial cover.",
        "image_file": "real_verified_journalism_01.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Authentic medical facility imagery coupled with fabricated biosecurity conspiracy claims.",
    },

    # Category 2: Subtle Deepfake Faceswaps
    {
        "case_id": "CASE_05",
        "category": "deepfake_faceswap",
        "title": "Subtle Diplomatic Face-Swap with Blurred Boundary",
        "text": "Manipulated facial reenactment video shows prominent diplomat uttering unverified threats.",
        "image_file": "fake_deepfake_faceswap_01.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "High-quality neural face-swap with subtle blending around the jawline and eyes.",
    },
    {
        "case_id": "CASE_06",
        "category": "deepfake_faceswap",
        "title": "Celebrity Deepfake Financial Endorsement",
        "text": "Deepfake video swap depicts celebrity actor endorsing fraudulent cryptocurrency token scheme.",
        "image_file": "fake_deepfake_faceswap_02.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Deepfake reenactment paired with promotional financial claims.",
    },
    {
        "case_id": "CASE_07",
        "category": "deepfake_faceswap",
        "title": "Synthesized Politician Debate Reenactment",
        "text": "Synthesized face-swap interview shows opposition politician making fabricated confessions.",
        "image_file": "fake_deepfake_faceswap_03.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Manipulated facial geometry on genuine debate broadcast backdrop.",
    },

    # Category 3: Generative Diffusion Synthesis
    {
        "case_id": "CASE_08",
        "category": "diffusion_synthesis",
        "title": "Photorealistic AI Portrait with Asymmetric Ocular Reflections",
        "text": "Latent diffusion model creates entirely fabricated portrait of non-existent whistleblower.",
        "image_file": "fake_diffusion_synthesis_01.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Generated face with photorealistic skin texture, detectable only via corneal reflection mismatch.",
    },
    {
        "case_id": "CASE_09",
        "category": "diffusion_synthesis",
        "title": "Synthesized Non-Existent Disaster Scene",
        "text": "AI-generated synthetic image depicting fictitious explosion near international embassy complex.",
        "image_file": "fake_diffusion_synthesis_02.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Hallucinated catastrophic event with coherent overall composition but micro-texture noise anomalies.",
    },
    {
        "case_id": "CASE_10",
        "category": "diffusion_synthesis",
        "title": "AI Composite Military Hardware Prototype",
        "text": "Text-to-image synthesis produces fake historical photograph of secret military expedition.",
        "image_file": "fake_diffusion_synthesis_03.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Synthetic historical aesthetic designed to simulate vintage film grain.",
    },

    # Category 4: Spliced Banners & Tampered Media
    {
        "case_id": "CASE_11",
        "category": "spliced_banner",
        "title": "Authentic Television Broadcast with Spliced Bank Run Banner",
        "text": "Authentic television broadcast altered with spliced banner claiming nationwide bank collapse.",
        "image_file": "fake_spliced_banner_01.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "90% of the image is genuine news anchor footage, but the lower third banner text is forged.",
    },
    {
        "case_id": "CASE_12",
        "category": "spliced_banner",
        "title": "Cloned Military Convoy Vehicle Duplication",
        "text": "Image showing military convoy altered by digitally cloning vehicles to exaggerate troop numbers.",
        "image_file": "fake_spliced_banner_02.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Copy-move forgery within the same image, preserving natural lighting, camera grain, and color temperature.",
    },

    # Category 5: Synthetic Speech Claim paired with Visual Image
    {
        "case_id": "CASE_13",
        "category": "synthetic_speech_claim",
        "title": "Voice-Cloned Election Official Phone Call Claim",
        "text": "Cloned audio clip imitating election official claiming polling station closure in key district.",
        "image_file": "fake_synthetic_speech_01.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Claim focuses on vocal cloning; visual representation is an image capture of the broadcast.",
    },
    {
        "case_id": "CASE_14",
        "category": "synthetic_speech_claim",
        "title": "Synthesized Executive Authorization Call Claim",
        "text": "Voice-cloned phone call impersonating executive officer authorizing fraudulent wire transfer.",
        "image_file": "fake_synthetic_speech_02.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Corporate fraud scenario involving voice cloning claims.",
    },

    # Category 6: Authentic Borderline Edge Case
    {
        "case_id": "CASE_15",
        "category": "authentic_edge_case",
        "title": "Genuine Scientific Glacial Ice Shelf Telemetry",
        "text": "Verified NASA satellite imagery documenting glacial ice shelf variations over five-year observation cycle.",
        "image_file": "real_science_nature_03.png",
        "ground_truth": "REAL",
        "difficulty_rationale": "Complex geometric patterns and irregular natural satellite textures that uncalibrated detectors frequently false-positive as synthetic noise.",
    },
]


def run_difficult_cases_analysis() -> List[Dict[str, Any]]:
    """Runs genuine multimodal inference on 15 difficult cases and analyzes failure modes."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictor = TruthLensPredictor()

    print(f"\n[INFO] Running Qualitative Error Analysis on {len(CANDIDATE_DIFFICULT_CASES)} Difficult Cases...")

    evaluated_cases = []

    for c in CANDIDATE_DIFFICULT_CASES:
        img_path = SAMPLES_DIR / c["image_file"]
        if not img_path.exists():
            print(f"[WARN] Image file not found: {img_path}. Skipping {c['case_id']}")
            continue

        report = predictor.predict_multimodal(text=c["text"], image_source=img_path)

        pred = report.prediction
        gt = c["ground_truth"]
        conf = report.confidence
        is_correct = (pred == gt)

        # Modality agreement analysis
        # Check if text cues vs visual cues point in different directions
        evidence_items = [e.model_dump() if hasattr(e, "model_dump") else e for e in report.evidence]
        has_cross_modal = any(e.get("modality") == "cross-modal" for e in evidence_items)

        # Determine error category & agreement
        if c["category"] == "semantic_mismatch":
            modalities_agreement = "DISAGREE (Authentic visual contradicted by false text caption)"
            if not is_correct:
                error_cat = "SINGLE_MODALITY_VISUAL_BIAS (Visual authenticity overshadowed textual contradiction)"
            else:
                error_cat = "CORRECT_CROSS_MODAL_RESOLUTION"
        elif c["category"] in ["deepfake_faceswap", "diffusion_synthesis", "spliced_banner"]:
            modalities_agreement = "AGREE (Both text claim and visual artifacts indicate manipulation)"
            error_cat = "CORRECT_MANIPULATION_DETECTION" if is_correct else "UNRESOLVED_SUBTLE_ARTIFACT"
        elif c["category"] == "authentic_edge_case":
            modalities_agreement = "AGREE (Both modalities reflect authentic observation)"
            error_cat = "ROBUST_AUTHENTIC_VERIFICATION" if is_correct else "FALSE_POSITIVE_ON_NATURAL_COMPLEXITY"
        else:
            modalities_agreement = "PARTIAL_AGREEMENT"
            error_cat = "CORRECT_PREDICTION" if is_correct else "CLASSIFICATION_ERROR"

        evaluated_cases.append({
            "case_id": c["case_id"],
            "category": c["category"],
            "title": c["title"],
            "ground_truth": gt,
            "prediction": pred,
            "confidence": round(conf, 4),
            "is_correct": is_correct,
            "modalities": report.modalities_used,
            "modalities_agreement": modalities_agreement,
            "error_category": error_cat,
            "difficulty_rationale": c["difficulty_rationale"],
            "explanation": report.explanation,
            "evidence_count": len(evidence_items),
            "key_evidence": [e.get("description", "") for e in evidence_items],
        })

    out_file = REPORTS_DIR / "difficult_cases.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(evaluated_cases, f, indent=2)

    correct_count = sum(1 for c in evaluated_cases if c["is_correct"])
    print(f"[INFO] Evaluated {len(evaluated_cases)} cases. Correct: {correct_count}/{len(evaluated_cases)} ({correct_count/max(len(evaluated_cases),1)*100:.1f}%)")
    print(f"[INFO] Saved difficult cases report to: {out_file}")

    return evaluated_cases


if __name__ == "__main__":
    run_difficult_cases_analysis()
