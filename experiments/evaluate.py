"""Master Evaluation Runner for TruthLens AI (Phase 4).

Orchestrates all experimental evaluation pipelines:
1. Dataset Audit (leakage, balance, integrity)
2. Unimodal Baselines (Text-only, Vision-only, Video technical check, Audio status)
3. Multimodal Fusion Comparison (Concat vs Bidirectional Cross-Attention)
4. Modality Ablation Study (Full vs -Text vs -Visual vs -Audio)
5. Qualitative Error Analysis on 15 Difficult Cases
6. Explainability & Evidence Pathway Evaluation
7. Chart and Visualizations Generation (Confusion Matrices, ROC, PR, Metrics Bar Chart)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.dataset_audit import audit_dataset
from experiments.baselines import run_all_baselines
from experiments.fusion_comparison import run_fusion_comparison
from experiments.ablation import run_ablation_study
from experiments.difficult_cases import run_difficult_cases_analysis
from experiments.explainability_eval import run_explainability_evaluation
from experiments.visualizations import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_pr_curves,
    plot_metrics_comparison,
)

REPORTS_DIR = BASE_DIR / "experiments" / "reports"
FIGURES_DIR = BASE_DIR / "reports" / "figures"


def run_master_evaluation(device_str: str = "auto", split: str = "test") -> Dict[str, Any]:
    """Executes the complete Phase 4 experimental evaluation suite."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("TRUTHLENS AI — PHASE 4 COMPREHENSIVE EXPERIMENTAL EVALUATION")
    print("=" * 70)

    # 1. Dataset Audit
    print("\n--- STAGE 1: DATASET AUDIT ---")
    audit_data = audit_dataset()

    # 2. Unimodal Baselines
    print("\n--- STAGE 2: UNIMODAL BASELINES ---")
    baselines_data = run_all_baselines(device_str=device_str, split=split)

    # 3. Fusion Comparison
    print("\n--- STAGE 3: FUSION STRATEGY COMPARISON ---")
    fusion_data = run_fusion_comparison(device_str=device_str, split=split)

    # 4. Modality Ablation
    print("\n--- STAGE 4: MODALITY ABLATION STUDY ---")
    ablation_data = run_ablation_study(device_str=device_str, split=split)

    # 5. Difficult Cases Analysis
    print("\n--- STAGE 5: 15 DIFFICULT CASES QUALITATIVE ANALYSIS ---")
    difficult_cases_data = run_difficult_cases_analysis()

    # 6. Explainability Evaluation
    print("\n--- STAGE 6: EXPLAINABILITY & EVIDENCE VERIFICATION ---")
    explainability_data = run_explainability_evaluation()

    # 7. Generate Visualizations from genuine model evaluations
    print("\n--- STAGE 7: PUBLISHABLE VISUALIZATIONS GENERATION ---")
    models_to_plot = {
        "Text Baseline": baselines_data["text_baseline"],
        "Vision Baseline": baselines_data["vision_baseline"],
        "Multimodal Concat": fusion_data["concatenation"],
        "Multimodal Cross-Attn": fusion_data["cross_attention"],
    }
    plot_confusion_matrices(models_to_plot, FIGURES_DIR)
    plot_roc_curves(models_to_plot, FIGURES_DIR)
    plot_pr_curves(models_to_plot, FIGURES_DIR)
    plot_metrics_comparison(models_to_plot, FIGURES_DIR)

    # Master consolidation
    master_results = {
        "evaluation_split": split,
        "dataset_audit": audit_data,
        "baselines": {
            k: {
                "experiment_id": v["experiment_id"],
                "model_name": v["model_name"],
                "status": v["status"],
                "metrics": v.get("metrics"),
                "reason": v.get("reason"),
            }
            for k, v in baselines_data.items()
        },
        "fusion_comparison": {
            k: {
                "experiment_id": v["experiment_id"],
                "fusion_type": v["fusion_type"],
                "metrics": v["metrics"],
            }
            for k, v in fusion_data.items()
        },
        "ablation_study": ablation_data,
        "difficult_cases_summary": {
            "total_evaluated": len(difficult_cases_data),
            "correct_count": sum(1 for c in difficult_cases_data if c["is_correct"]),
            "misclassified_count": sum(1 for c in difficult_cases_data if not c["is_correct"]),
            "accuracy": round(sum(1 for c in difficult_cases_data if c["is_correct"]) / max(len(difficult_cases_data), 1), 4),
        },
        "explainability_evaluation": explainability_data,
    }

    out_file = REPORTS_DIR / "master_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 4 EVALUATION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Experiment':<25} | {'Modalities':<12} | {'Fusion':<14} | {'Accuracy':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 85)

    # Baselines
    for k in ["text_baseline", "vision_baseline"]:
        b = baselines_data[k]
        m = b["metrics"]
        print(f"{b['experiment_id']:<25} | {b['modality']:<12} | {'N/A':<14} | {m['accuracy']*100:.2f}%  | {m['f1']*100:.2f}%  | {m['roc_auc'] if m['roc_auc'] is not None else 'N/A'}")

    for k in ["video_baseline", "audio_baseline"]:
        b = baselines_data[k]
        print(f"{b['experiment_id']:<25} | {b['modality']:<12} | {'N/A':<14} | {'N/A':<8} | {'N/A':<8} | {b['status']}")

    # Fusion Comparison
    for k in ["concatenation", "cross_attention"]:
        f_res = fusion_data[k]
        m = f_res["metrics"]
        print(f"{f_res['experiment_id']:<25} | {f_res['modalities']:<12} | {f_res['fusion_type']:<14} | {m['accuracy']*100:.2f}%  | {m['f1']*100:.2f}%  | {m['roc_auc'] if m['roc_auc'] is not None else 'N/A'}")

    print("=" * 70 + "\n")
    print(f"[INFO] Master evaluation report saved to: {out_file}")
    return master_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthLens Master Evaluation Runner")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    run_master_evaluation(device_str=args.device, split=args.split)
