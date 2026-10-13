"""Terminal demonstration script showcasing multimodal deepfake verification with explainability."""

import os
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.inference.predictor import TruthLensPredictor


def run_terminal_demo(sample_id: str = "sample_002"):
    print("=" * 70)
    print("TRUTHLENS AI - MULTIMODAL VERIFICATION CLI DEMO")
    print("=" * 70)

    predictor = TruthLensPredictor.get_instance()

    # Locate sample
    meta_path = BASE_DIR / "ml" / "data" / "metadata" / "dataset.csv"
    if not meta_path.exists():
        print("[INFO] Generating development samples first...")
        from scripts.download_dataset import create_development_samples
        create_development_samples()

    # Find sample in CSV
    import csv
    chosen_row = None
    with open(meta_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["id"] == sample_id or sample_id in r["image_path"]:
                chosen_row = r
                break

    if not chosen_row:
        # Fallback to first available sample
        with open(meta_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            chosen_row = next(reader)

    print(f"\n[INPUT DATA]")
    print(f"Sample ID:   {chosen_row['id']}")
    print(f"Claim/Text:  '{chosen_row['text']}'")
    print(f"Image Path:  {chosen_row['image_path']}")
    print(f"True Label:  {'FAKE (1)' if chosen_row['label'] == '1' else 'REAL (0)'}")

    full_img_path = BASE_DIR / "ml" / "data" / chosen_row["image_path"]

    print("\n[RUNNING MULTIMODAL PIPELINE]...")
    report = predictor.predict_multimodal(
        text=chosen_row["text"],
        image_source=full_img_path,
    )

    print("\n" + "=" * 70)
    print("TRUTHLENS AI VERIFICATION REPORT")
    print("=" * 70)
    print(f"Prediction:          {report.prediction}")
    print(f"Confidence:          {report.confidence * 100:.1f}%")
    print(f"Real Probability:    {report.real_probability * 100:.1f}%")
    print(f"Fake Probability:    {report.fake_probability * 100:.1f}%")
    print(f"Uncertain Flag:      {report.uncertain}")
    print(f"Manipulation Type:   {report.manipulation_type}")
    print(f"Modalities Analyzed: {', '.join(report.modalities_used)}")
    print(f"Model Architecture:  {report.model_version}")
    print(f"Compute Device:      {report.device_used}")
    print(f"\n[EXPLANATION]")
    print(report.explanation)
    print(f"\n[EVIDENCE DETAILS]")
    for idx, ev in enumerate(report.evidence, 1):
        print(f"{idx}. [{ev.modality.upper()}] ({ev.type}): {ev.description}")
        if ev.token_importance:
            tokens_str = ", ".join([f"{t.token}:{t.score:.2f}" for t in ev.token_importance[:5]])
            print(f"   Top Influential Tokens: {tokens_str}")
        if ev.heatmap_available:
            print("   Grad-CAM Heatmap: Generated and available for visual inspection.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TruthLens AI Demo")
    parser.add_argument("--sample", type=str, default="sample_002", help="Sample ID (e.g. sample_001 or sample_002)")
    args = parser.parse_args()
    run_terminal_demo(args.sample)
