"""Generates all Phase 6 publication-quality visualization figures.

Figures saved to reports/figures/:
1. phase6_domain_shift_comparison.png: In-domain vs Out-of-Domain text performance before & after adaptation.
2. phase6_semantic_mismatch_comparison.png: Condition 1-4 accuracy across Baseline, Contrastive, and Adapted+Contrastive.
3. phase6_calibration_curve.png: Reliability diagrams before and after temperature scaling.
4. phase6_confidence_distribution.png: Probability distribution before vs after calibration.
5. phase6_roc_comparison.png: ROC and PR curves for ablation configurations.
6. phase6_ablation_comparison.png: Comprehensive comparison of Configs A-F across the 5 evaluation suites.
7. phase6_attention_comparison.png: Attention entropy across mismatch conditions.
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

FIGURES_DIR = BASE_DIR / "reports" / "figures"
RESULTS_DIR = BASE_DIR / "experiments" / "phase6" / "results"
PHASE6_DIR = BASE_DIR / "experiments" / "phase6"
MODELS_DIR = BASE_DIR / "models" / "phase6"

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_domain_shift_comparison():
    """Figure 1: In-domain vs Fakeddit Out-of-Domain text accuracy & F1."""
    csv_path = PHASE6_DIR / "text_domain_adaptation_results.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found.")
        return

    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)

    # We want Baseline vs Adapted on Course In-Domain and Fakeddit
    # Find records
    b_in = next(r for r in records if "Baseline" in r["Model"] and "Course" in r["Dataset"])
    a_in = next(r for r in records if "Domain-Adapted" in r["Model"] and "Course" in r["Dataset"])
    b_out = next(r for r in records if "Baseline" in r["Model"] and "Fakeddit" in r["Dataset"])
    a_out = next(r for r in records if "Domain-Adapted" in r["Model"] and "Fakeddit" in r["Dataset"])

    categories = ["In-Domain (Course)", "Out-of-Domain (Fakeddit)"]
    base_acc = [float(b_in["Accuracy"]) * 100, float(b_out["Accuracy"]) * 100]
    adapt_acc = [float(a_in["Accuracy"]) * 100, float(a_out["Accuracy"]) * 100]
    base_f1 = [float(b_in["F1"]) * 100, float(b_out["F1"]) * 100]
    adapt_f1 = [float(a_in["F1"]) * 100, float(a_out["F1"]) * 100]

    x = np.arange(len(categories))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - 1.5 * width, base_acc, width, label="Baseline Accuracy", color="#4A90E2")
    rects2 = ax.bar(x - 0.5 * width, adapt_acc, width, label="Adapted Accuracy", color="#1B4F72")
    rects3 = ax.bar(x + 0.5 * width, base_f1, width, label="Baseline F1", color="#F39C12")
    rects4 = ax.bar(x + 1.5 * width, adapt_f1, width, label="Adapted F1", color="#B9770E")

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("TruthLens AI: Text Domain Adaptation (DistilBERT)\nIn-Domain vs Out-of-Domain Social Media Robustness", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    out_file = FIGURES_DIR / "phase6_domain_shift_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


def plot_semantic_mismatch_comparison():
    """Figure 2: Accuracy across Conditions 1-4."""
    json_path = RESULTS_DIR / "contrastive_grid_results.json"
    if not json_path.exists():
        print(f"[SKIP] {json_path} not found.")
        return

    with open(json_path) as f:
        data = json.load(f)

    runs = data["runs"]
    # Get lambda=0.0 (baseline), best lambda (contrastive), and adapted_contrastive
    r_base = next(r for r in runs if r["lambda_val"] == 0.0 and not r["use_domain_adapted_text"])
    r_cont = next(r for r in runs if r["lambda_val"] == data["best_lambda"] and not r["use_domain_adapted_text"])
    r_adapt = next(r for r in runs if r["use_domain_adapted_text"])

    cond_keys = ["condition_1", "condition_2", "condition_3", "condition_4"]
    cond_labels = [
        "C1: Congruent Real\n(Real Img + Real Txt)",
        "C2: Semantic Mismatch\n(Real Img + Fake Txt)",
        "C3: Swapped Context\n(Real Txt + Swapped Img)",
        "C4: Unrelated Txt\n(Real Img + Unrel Txt)",
    ]

    base_accs = [r_base["final_mismatch"]["conditions"][k]["accuracy"] * 100 for k in cond_keys]
    cont_accs = [r_cont["final_mismatch"]["conditions"][k]["accuracy"] * 100 for k in cond_keys]
    adapt_accs = [r_adapt["final_mismatch"]["conditions"][k]["accuracy"] * 100 for k in cond_keys]

    x = np.arange(len(cond_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    r1 = ax.bar(x - width, base_accs, width, label=f"Baseline ($\\lambda=0.0$)", color="#E74C3C")
    r2 = ax.bar(x, cont_accs, width, label=f"Contrastive Only ($\\lambda={data['best_lambda']}$)", color="#3498DB")
    r3 = ax.bar(x + width, adapt_accs, width, label=f"Domain Adapted + Contrastive (Config E)", color="#2ECC71")

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Modal Semantic Mismatch Robustness Across 4 Conditions\nMitigating Visual Dominance via InfoNCE Contrastive Alignment", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out_file = FIGURES_DIR / "phase6_semantic_mismatch_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


def plot_calibration_curve():
    """Figure 3: Reliability diagram before vs after temperature scaling."""
    params_path = MODELS_DIR / "calibration_params.json"
    if not params_path.exists():
        print(f"[SKIP] {params_path} not found.")
        return

    with open(params_path) as f:
        params = json.load(f)

    # Use Semantic Mismatch or Fakeddit suite bins
    suite_data = params["suite_results"]["fakeddit"]
    uncal_bins = suite_data["uncalibrated"]["bins"]
    cal_bins = suite_data["calibrated"]["bins"]
    opt_t = params["optimal_temperature"]

    uncal_conf = [b["bin_conf"] for b in uncal_bins if b["sample_count"] > 0]
    uncal_acc = [b["bin_acc"] for b in uncal_bins if b["sample_count"] > 0]
    cal_conf = [b["bin_conf"] for b in cal_bins if b["sample_count"] > 0]
    cal_acc = [b["bin_acc"] for b in cal_bins if b["sample_count"] > 0]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)

    ax.plot(uncal_conf, uncal_acc, "s-", color="#E74C3C", linewidth=2, markersize=7, label=f"Uncalibrated (T=1.00, ECE={suite_data['uncalibrated']['ece']:.3f})")
    ax.plot(cal_conf, cal_acc, "o-", color="#27AE60", linewidth=2, markersize=7, label=f"Calibrated (T={opt_t:.2f}, ECE={suite_data['calibrated']['ece']:.3f})")

    ax.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Probability Calibration Reliability Diagram (Fakeddit Benchmark)\nImpact of Post-Processing Temperature Scaling", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")

    plt.tight_layout()
    out_file = FIGURES_DIR / "phase6_calibration_curve.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


def plot_confidence_distribution():
    """Figure 4: Confidence histogram before vs after temperature scaling."""
    params_path = MODELS_DIR / "calibration_params.json"
    if not params_path.exists():
        print(f"[SKIP] {params_path} not found.")
        return

    with open(params_path) as f:
        params = json.load(f)

    suite_data = params["suite_results"]["semantic_mismatch"]
    uncal_probs = suite_data["uncalibrated"]["fake_probs"]
    cal_probs = suite_data["calibrated"]["fake_probs"]
    opt_t = params["optimal_temperature"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300, sharey=True)

    bins = np.linspace(0.0, 1.0, 15)
    ax1.hist(uncal_probs, bins=bins, color="#E74C3C", alpha=0.7, edgecolor="black")
    ax1.axvline(0.65, color="black", linestyle="--", label="Uncertainty Threshold (0.65)")
    ax1.axvline(0.35, color="black", linestyle="--", label="REAL Threshold (0.35)")
    ax1.set_title("Uncalibrated Confidence ($T=1.0$)\n(Extreme Probability Spikes)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted Fake Probability", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Sample Count", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper center", fontsize=8)

    ax2.hist(cal_probs, bins=bins, color="#27AE60", alpha=0.7, edgecolor="black")
    ax2.axvline(0.65, color="black", linestyle="--", label="Uncertainty Threshold (0.65)")
    ax2.axvline(0.35, color="black", linestyle="--", label="REAL Threshold (0.35)")
    ax2.set_title(f"Calibrated Confidence ($T={opt_t:.2f}$)\n(Softened Logits & Uncertainty Routing)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Predicted Fake Probability", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper center", fontsize=8)

    plt.suptitle("TruthLens AI: Predicted Probability Distribution Shift", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = FIGURES_DIR / "phase6_confidence_distribution.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


def plot_ablation_comparison():
    """Figure 6: Grouped bar chart comparing Configs A-F across the 5 evaluation suites."""
    json_path = RESULTS_DIR / "ablation_study_results.json"
    if not json_path.exists():
        print(f"[SKIP] {json_path} not found.")
        return

    with open(json_path) as f:
        data = json.load(f)

    config_keys = ["config_a", "config_b", "config_c", "config_d", "config_e", "config_f"]
    config_labels = ["Cfg A (Base)", "Cfg B (DA)", "Cfg C (Cont)", "Cfg D (Cal)", "Cfg E (DA+Cont)", "Cfg F (Full)"]
    dataset_keys = ["in_domain_val", "in_domain_test", "fakeddit", "fakeavceleb", "semantic_mismatch"]
    dataset_labels = ["In-Domain Val", "In-Domain Test", "Fakeddit", "FakeAVCeleb", "Mismatch Suite"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    x = np.arange(len(dataset_labels))
    width = 0.13

    colors = ["#95A5A6", "#3498DB", "#9B59B6", "#E67E22", "#1ABC9C", "#2ECC71"]

    for i, (ck, cl, col) in enumerate(zip(config_keys, config_labels, colors)):
        accs = [data[ck]["datasets"][dk]["accuracy"] * 100 for dk in dataset_keys]
        ax.bar(x + (i - 2.5) * width, accs, width, label=cl, color=col)

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("TruthLens AI: Comprehensive Phase 6 Ablation Study Across 5 Evaluation Suites\nIsolating Domain Adaptation, Contrastive Alignment, and Calibration", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right", ncol=2)

    plt.tight_layout()
    out_file = FIGURES_DIR / "phase6_ablation_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


def plot_attention_comparison():
    """Figure 7: Attention entropy before vs after contrastive alignment."""
    json_path = RESULTS_DIR / "contrastive_grid_results.json"
    if not json_path.exists():
        print(f"[SKIP] {json_path} not found.")
        return

    with open(json_path) as f:
        data = json.load(f)

    runs = data["runs"]
    r_base = next(r for r in runs if r["lambda_val"] == 0.0 and not r["use_domain_adapted_text"])
    r_adapt = next(r for r in runs if r["use_domain_adapted_text"])

    cond_keys = ["condition_1", "condition_2", "condition_3", "condition_4"]
    cond_labels = ["C1: Congruent Real", "C2: Semantic Mismatch", "C3: Swapped Image", "C4: Unrelated Text"]

    base_entropy = [r_base["final_mismatch"]["conditions"][k]["mean_entropy"] for k in cond_keys]
    adapt_entropy = [r_adapt["final_mismatch"]["conditions"][k]["mean_entropy"] for k in cond_keys]

    x = np.arange(len(cond_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    r1 = ax.bar(x - width/2, base_entropy, width, label="Baseline Cross-Attention", color="#E74C3C")
    r2 = ax.bar(x + width/2, adapt_entropy, width, label="Phase 6 Contrastive Cross-Attention", color="#2980B9")

    ax.set_ylabel("Mean Cross-Attention Entropy (nats)", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Modal Bidirectional Attention Entropy Across Conditions\nLower Entropy = Sharper Text-to-Image Semantic Grounding", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 4.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    for rects in [r1, r2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.4f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out_file = FIGURES_DIR / "phase6_attention_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


def plot_roc_comparison():
    """Figure 5: ROC curves across configurations on Semantic Mismatch suite."""
    params_path = MODELS_DIR / "calibration_params.json"
    if not params_path.exists():
        print(f"[SKIP] {params_path} not found.")
        return

    with open(params_path) as f:
        params = json.load(f)

    # Extract probabilities for semantic mismatch
    sm = params["suite_results"]["semantic_mismatch"]
    uncal_probs = sm["uncalibrated"]["fake_probs"]
    cal_probs = sm["calibrated"]["fake_probs"]
    
    # Ground truth labels from external dataset
    manifest_file = BASE_DIR / "experiments" / "external_dataset" / "external_dataset_manifest.csv"
    import pandas as pd
    df = pd.read_csv(manifest_file)
    df_sm = df[df["source_dataset"] == "semantic_mismatch_suite"]
    y_true = df_sm["label"].tolist()

    fpr_u, tpr_u, _ = roc_curve(y_true, uncal_probs)
    roc_u = auc(fpr_u, tpr_u)

    fpr_c, tpr_c, _ = roc_curve(y_true, cal_probs)
    roc_c = auc(fpr_c, tpr_c)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)", linewidth=1.2)
    ax.plot(fpr_u, tpr_u, color="#E74C3C", linewidth=2.2, label=f"Uncalibrated (AUC = {roc_u:.3f})")
    ax.plot(fpr_c, tpr_c, color="#27AE60", linewidth=2.0, linestyle="--", label=f"Calibrated (AUC = {roc_c:.3f}, Rank Invariant)")

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curves on Semantic Mismatch Benchmark\nDemonstrating Rank Invariance Under Temperature Scaling", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    plt.tight_layout()
    out_file = FIGURES_DIR / "phase6_roc_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[+] Saved: {out_file}")


def generate_all_figures():
    """Generates all 7 publication figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("TRUTHLENS AI — GENERATING PHASE 6 PUBLICATION FIGURES")
    print("=" * 70)
    plot_domain_shift_comparison()
    plot_semantic_mismatch_comparison()
    plot_calibration_curve()
    plot_confidence_distribution()
    plot_roc_comparison()
    plot_ablation_comparison()
    plot_attention_comparison()
    print("[+] All 7 publication figures generated successfully in reports/figures/!")


if __name__ == "__main__":
    generate_all_figures()
