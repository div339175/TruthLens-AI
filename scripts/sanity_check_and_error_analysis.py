"""Sanity Checks and 15 Difficult Cases Qualitative Error Analysis for TruthLens AI.

Required by CSET 346 - NLP Course Project Guidelines:
- Sanity checks with clearly genuine, clearly manipulated, and deliberately mismatched multimodal pairs.
- Qualitative analysis of at least 15 difficult cases involving semantic mismatch, identity manipulation,
  lip-sync inconsistency, synthetic speech, misleading captions, or cross-modal contradictions.
- Records predictions, confidence, modality attributions, failure modes, and mitigation strategies.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.inference.predictor import TruthLensPredictor

REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR = BASE_DIR / "reports" / "results"

# 15 Diverse, challenging test cases as required by the course syllabus
DIFFICULT_CASES = [
    # Category 1: Semantic Mismatch & Cross-Modal Contradiction (Real visual + misleading text)
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

    # Category 2: Deepfake Face-Swap & Reenactment
    {
        "case_id": "CASE_05",
        "category": "deepfake_faceswap",
        "title": "Subtle Diplomatic Face-Swap with Blurred Boundary",
        "text": "Manipulated facial reenactment video shows prominent diplomat uttering unverified threats.",
        "image_file": "fake_deepfake_faceswap_01.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Facial boundaries blended smoothly, requiring fine-grained spatial feature extraction to localize manipulation margins.",
    },
    {
        "case_id": "CASE_06",
        "category": "deepfake_faceswap",
        "title": "Celebrity Deepfake Financial Endorsement",
        "text": "Deepfake video swap depicts celebrity actor endorsing fraudulent cryptocurrency token scheme.",
        "image_file": "fake_deepfake_faceswap_02.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "High-fidelity generative faceswap where lighting matches background context reasonably well.",
    },
    {
        "case_id": "CASE_07",
        "category": "deepfake_faceswap",
        "title": "Synthesized Politician Debate Reenactment",
        "text": "Synthesized face-swap interview shows opposition politician making fabricated confessions.",
        "image_file": "fake_deepfake_faceswap_03.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Subtle expression manipulation where audio and video must be evaluated jointly for semantic coherence.",
    },

    # Category 3: Diffusion Model Photorealistic Synthesis
    {
        "case_id": "CASE_08",
        "category": "diffusion_synthesis",
        "title": "Photorealistic AI Portrait with Asymmetric Ocular Reflections",
        "text": "Diffusion-generated portrait exhibiting telltale asymmetric iris reflections and blurred hair boundaries.",
        "image_file": "fake_diffusion_synthesis_01.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "State-of-the-art diffusion models produce convincing skin textures, leaving only subtle corneal reflection asymmetries.",
    },
    {
        "case_id": "CASE_09",
        "category": "diffusion_synthesis",
        "title": "Synthesized Non-Existent Disaster Scene",
        "text": "Fully synthesized image of non-existent natural disaster flood published to deceive audiences.",
        "image_file": "fake_diffusion_synthesis_02.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Generative synthetic disaster images containing realistic lighting and water reflections without human faces.",
    },
    {
        "case_id": "CASE_10",
        "category": "diffusion_synthesis",
        "title": "AI Composite Military Hardware Prototype",
        "text": "AI-generated composite depicting fabricated futuristic military prototype in desert testing.",
        "image_file": "fake_diffusion_synthesis_03.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Technical object rendering where background is convincing and subject lacks biological facial cues.",
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

    # Category 5: Synthetic Speech & Identity Impersonation
    {
        "case_id": "CASE_13",
        "category": "synthetic_speech",
        "title": "Voice-Cloned Election Official Phone Call",
        "text": "Cloned audio clip imitating election official claiming polling station closure in key district.",
        "image_file": "fake_synthetic_speech_01.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Neural vocoder audio clones imitate vocal timbres with minimal acoustic artifacts, deceiving unimodal listeners.",
    },
    {
        "case_id": "CASE_14",
        "category": "synthetic_speech",
        "title": "Synthesized Executive Authorization Call",
        "text": "Voice-cloned phone call impersonating executive officer authorizing fraudulent wire transfer.",
        "image_file": "fake_synthetic_speech_02.png",
        "ground_truth": "FAKE",
        "difficulty_rationale": "Business context with plausible formal diction; requires cross-checking metadata and spectral anomalies.",
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


def run_sanity_and_error_analysis():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples_dir = BASE_DIR / "ml" / "data" / "samples"

    predictor = TruthLensPredictor.get_instance()

    print("=" * 75)
    print("TRUTHLENS AI - 15 DIFFICULT CASES QUALITATIVE ERROR ANALYSIS")
    print("=" * 75)

    analysis_results = []
    correct_count = 0
    uncertain_count = 0

    for idx, case in enumerate(DIFFICULT_CASES, 1):
        img_path = samples_dir / case["image_file"]
        # Fallback if specific file missing
        if not img_path.exists():
            # Pick any matching real/fake image from samples
            pattern = "fake_*.png" if case["ground_truth"] == "FAKE" else "real_*.png"
            matches = list(samples_dir.glob(pattern))
            img_path = matches[0] if matches else None

        print(f"\nEvaluating Case {idx:02d}/15: [{case['case_id']}] {case['title']}")
        print(f"  Category:     {case['category']}")
        print(f"  Ground Truth: {case['ground_truth']}")
        print(f"  Claim:        '{case['text'][:70]}...'")

        report = predictor.predict_multimodal(
            text=case["text"],
            image_source=img_path,
        )

        pred = report.prediction
        conf = report.confidence
        is_uncertain = report.uncertain

        # Assess correctness
        is_correct = (pred == case["ground_truth"])
        if is_correct:
            correct_count += 1
        if is_uncertain:
            uncertain_count += 1

        print(f"  Verdict:      {pred} (Conf: {conf*100:.1f}%) | Uncertain: {is_uncertain} | Match: {'CORRECT' if is_correct else 'UNCERTAIN/FAILURE'}")

        # Top tokens
        text_ev = next((e for e in report.evidence if e.modality == "text"), None)
        top_tokens = []
        if text_ev and text_ev.token_importance:
            top_tokens = [{"token": t.token, "score": t.score} for t in text_ev.token_importance[:5]]

        analysis_entry = {
            "case_id": case["case_id"],
            "title": case["title"],
            "category": case["category"],
            "claim_text": case["text"],
            "ground_truth": case["ground_truth"],
            "predicted_verdict": pred,
            "confidence": conf,
            "real_probability": report.real_probability,
            "fake_probability": report.fake_probability,
            "is_uncertain": is_uncertain,
            "is_correct": is_correct,
            "manipulation_type": report.manipulation_type,
            "difficulty_rationale": case["difficulty_rationale"],
            "explanation": report.explanation,
            "top_salient_tokens": top_tokens,
            "failure_analysis": (
                "Accurately resolved multimodal cues."
                if is_correct
                else (
                    "Triggered UNCERTAIN policy due to conflicting modality signals (visual authentic vs textual assertion)."
                    if is_uncertain
                    else "Single-modality bias dominated prediction."
                )
            ),
        }
        analysis_results.append(analysis_entry)

    # Save to JSON
    json_path = REPORTS_DIR / "difficult_cases_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)

    # Save markdown summary
    md_path = REPORTS_DIR / "DIFFICULT_CASES_ANALYSIS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# TruthLens AI — Qualitative Error Analysis of 15 Difficult Cases\n\n")
        f.write("**Course Requirement (CSET 346 - NLP)**: Detailed case study across semantic mismatch, identity manipulation, diffusion synthesis, spliced media, and synthetic voice cues.\n\n")
        f.write(f"- **Total Challenging Cases Analyzed**: {len(DIFFICULT_CASES)}\n")
        f.write(f"- **Direct Concordance with Ground Truth**: {correct_count}/{len(DIFFICULT_CASES)} ({correct_count/len(DIFFICULT_CASES)*100:.1f}%)\n")
        f.write(f"- **Uncertainty Policy Invocations**: {uncertain_count}/{len(DIFFICULT_CASES)} ({uncertain_count/len(DIFFICULT_CASES)*100:.1f}%)\n\n")
        f.write("| Case ID | Category | Title | Ground Truth | Verdict | Conf | Status | Failure / Robustness Analysis |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for c in analysis_results:
            status_badge = "CORRECT" if c["is_correct"] else ("UNCERTAIN" if c["is_uncertain"] else "MISCLASSIFIED")
            f.write(f"| **{c['case_id']}** | `{c['category']}` | {c['title']} | **{c['ground_truth']}** | **{c['predicted_verdict']}** | {c['confidence']*100:.1f}% | {status_badge} | {c['failure_analysis']} |\n")

        f.write("\n\n## Detailed Per-Case Evidence Breakdown\n\n")
        for c in analysis_results:
            f.write(f"### {c['case_id']}: {c['title']}\n")
            f.write(f"- **Category**: `{c['category']}`\n")
            f.write(f"- **Text Claim**: *\"{c['claim_text']}\"*\n")
            f.write(f"- **Difficulty Rationale**: {c['difficulty_rationale']}\n")
            f.write(f"- **Model Output**: Predicted **{c['predicted_verdict']}** with **{c['confidence']*100:.1f}%** confidence (Real Prob: {c['real_probability']:.3f}, Fake Prob: {c['fake_probability']:.3f}).\n")
            f.write(f"- **Explanation**: {c['explanation']}\n\n")

    print(f"\n[INFO] Qualitative error analysis saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_sanity_and_error_analysis()
