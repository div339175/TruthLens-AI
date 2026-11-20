"""Modality Ablation Study for TruthLens AI (Phase 4).

Compares:
1. Full Model (Text + Visual, Cross-Attention)
2. Full - Text (Visual-only)
3. Full - Visual (Text-only)
4. Full - Audio (Documented as NOT AVAILABLE due to 0 labeled audio samples in dataset)

Calculates performance changes relative to the full model:
ΔF1 = F1_without_modality - F1_full
"""

import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.baselines import evaluate_text_baseline, evaluate_vision_baseline
from experiments.fusion_comparison import evaluate_multimodal_variant

REPORTS_DIR = BASE_DIR / "experiments" / "reports"
MODELS_DIR = BASE_DIR / "models"


def run_ablation_study(device_str: str = "auto", split: str = "test") -> Dict[str, Any]:
    """Runs systematic modality ablation study."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)

    print(f"\n[INFO] Running Phase 4 Modality Ablation on device={device}, split={split}...")

    # 1. Full Multimodal Model (Text + Visual, Cross-Attention)
    full_res = evaluate_multimodal_variant(
        experiment_id="EXP_ABLATION_FULL",
        fusion_type="cross_attention",
        checkpoint_file=MODELS_DIR / "multimodal_cross_attention.pt",
        device=device,
        split=split,
    )
    full_f1 = full_res["metrics"]["f1"]
    full_acc = full_res["metrics"]["accuracy"]

    # 2. Full - Text (Vision-Only)
    no_text_res = evaluate_vision_baseline(device, split=split)
    no_text_res["experiment_id"] = "EXP_ABLATION_NO_TEXT"
    no_text_f1 = no_text_res["metrics"]["f1"]
    no_text_acc = no_text_res["metrics"]["accuracy"]

    # 3. Full - Visual (Text-Only)
    no_vis_res = evaluate_text_baseline(device, split=split)
    no_vis_res["experiment_id"] = "EXP_ABLATION_NO_VISUAL"
    no_vis_f1 = no_vis_res["metrics"]["f1"]
    no_vis_acc = no_vis_res["metrics"]["accuracy"]

    # 4. Full - Audio
    no_audio_res = {
        "experiment_id": "EXP_ABLATION_NO_AUDIO",
        "model_name": "Full Model without Audio",
        "status": "NOT_AVAILABLE",
        "statement": "Audio ablation could not be quantitatively evaluated because the repository contains no labeled audio evaluation samples.",
        "delta_f1": None,
        "delta_acc": None,
        "metrics": None,
    }

    # Calculate relative changes (Delta)
    delta_no_text_f1 = round(no_text_f1 - full_f1, 4)
    delta_no_text_acc = round(no_text_acc - full_acc, 4)

    delta_no_vis_f1 = round(no_vis_f1 - full_f1, 4)
    delta_no_vis_acc = round(no_vis_acc - full_acc, 4)

    ablation_results = {
        "full_model": {
            "variant": "Full Model (Text + Visual, Cross-Attention)",
            "metrics": full_res["metrics"],
            "delta_f1": 0.0,
            "delta_acc": 0.0,
        },
        "without_text": {
            "variant": "Full - Text (Vision-Only)",
            "metrics": no_text_res["metrics"],
            "delta_f1": delta_no_text_f1,
            "delta_acc": delta_no_text_acc,
        },
        "without_visual": {
            "variant": "Full - Visual (Text-Only)",
            "metrics": no_vis_res["metrics"],
            "delta_f1": delta_no_vis_f1,
            "delta_acc": delta_no_vis_acc,
        },
        "without_audio": {
            "variant": "Full - Audio",
            "status": "NOT_AVAILABLE",
            "statement": "Audio ablation could not be quantitatively evaluated because the repository contains no labeled audio evaluation samples.",
            "metrics": None,
            "delta_f1": None,
            "delta_acc": None,
        },
    }

    # Save JSON report
    out_json = REPORTS_DIR / "ablation_study.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # Save CSV table
    out_csv = REPORTS_DIR / "ablation_study.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Configuration", "Accuracy", "F1-Score", "ΔAccuracy", "ΔF1-Score", "Status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({
            "Configuration": "Full Model (Text + Visual)",
            "Accuracy": f"{full_acc:.4f}",
            "F1-Score": f"{full_f1:.4f}",
            "ΔAccuracy": "0.0000 (Baseline)",
            "ΔF1-Score": "0.0000 (Baseline)",
            "Status": "COMPLETE",
        })
        writer.writerow({
            "Configuration": "Full - Text (Vision-Only)",
            "Accuracy": f"{no_text_acc:.4f}",
            "F1-Score": f"{no_text_f1:.4f}",
            "ΔAccuracy": f"{delta_no_text_acc:+.4f}",
            "ΔF1-Score": f"{delta_no_text_f1:+.4f}",
            "Status": "COMPLETE",
        })
        writer.writerow({
            "Configuration": "Full - Visual (Text-Only)",
            "Accuracy": f"{no_vis_acc:.4f}",
            "F1-Score": f"{no_vis_f1:.4f}",
            "ΔAccuracy": f"{delta_no_vis_acc:+.4f}",
            "ΔF1-Score": f"{delta_no_vis_f1:+.4f}",
            "Status": "COMPLETE",
        })
        writer.writerow({
            "Configuration": "Full - Audio",
            "Accuracy": "N/A",
            "F1-Score": "N/A",
            "ΔAccuracy": "N/A",
            "ΔF1-Score": "N/A",
            "Status": "NOT AVAILABLE (0 labeled audio samples in dataset)",
        })

    print(f"[INFO] Modality ablation study complete. Output saved to {out_json} and {out_csv}")
    return ablation_results


if __name__ == "__main__":
    run_ablation_study()
