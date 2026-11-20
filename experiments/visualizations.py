"""Visualizations generator for TruthLens AI (Phase 4).

Generates publishable figures from actual experimental evaluations:
- Confusion matrices for all supported models
- ROC curves with computed AUC scores
- Precision-Recall curves with computed PR-AUC scores
- Comparative performance bar chart across models
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"


def plot_confusion_matrices(models_results: Dict[str, Dict[str, Any]], out_dir: Path = FIGURES_DIR):
    """Plots confusion matrices in a side-by-side grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    num_models = len(models_results)
    if num_models == 0:
        return

    cols = 2
    rows = (num_models + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, data) in enumerate(models_results.items()):
        ax = axes[idx]
        cm = data["metrics"]["confusion_matrix"]["matrix"]
        cm_arr = np.array(cm)

        im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{name}\nAcc: {data['metrics']['accuracy']:.4f} | F1: {data['metrics']['f1']:.4f}", fontsize=11, fontweight="bold")

        classes = ["REAL", "FAKE"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=10)

        thresh = cm_arr.max() / 2.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(
                    j, i, format(cm_arr[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold",
                )

        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

    # Hide unused subplots if any
    for j in range(num_models, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    out_file = out_dir / "confusion_matrices.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confusion matrices to: {out_file}")


def plot_roc_curves(models_results: Dict[str, Dict[str, Any]], out_dir: Path = FIGURES_DIR):
    """Plots comparative ROC curves for models with probabilities."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    colors = ["#2b6cb0", "#c53030", "#2f855a", "#6b46c1"]

    for idx, (name, data) in enumerate(models_results.items()):
        raw = data.get("raw_predictions", {})
        y_true = raw.get("y_true", [])
        y_prob = raw.get("y_prob", [])

        if len(y_true) > 0 and len(y_prob) == len(y_true) and len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = data["metrics"].get("roc_auc", 0.0)
            color = colors[idx % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("TruthLens AI — ROC Curves on Test Split", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    out_file = out_dir / "roc_curves.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved ROC curves to: {out_file}")


def plot_pr_curves(models_results: Dict[str, Dict[str, Any]], out_dir: Path = FIGURES_DIR):
    """Plots comparative Precision-Recall curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    colors = ["#2b6cb0", "#c53030", "#2f855a", "#6b46c1"]

    for idx, (name, data) in enumerate(models_results.items()):
        raw = data.get("raw_predictions", {})
        y_true = raw.get("y_true", [])
        y_prob = raw.get("y_prob", [])

        if len(y_true) > 0 and len(y_prob) == len(y_true) and len(set(y_true)) > 1:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            pr_auc_val = data["metrics"].get("pr_auc", 0.0)
            color = colors[idx % len(colors)]
            plt.plot(rec, prec, color=color, lw=2.2, label=f"{name} (PR-AUC = {pr_auc_val:.4f})")

    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("TruthLens AI — Precision-Recall Curves on Test Split", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)

    out_file = out_dir / "pr_curves.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved Precision-Recall curves to: {out_file}")


def plot_metrics_comparison(models_results: Dict[str, Dict[str, Any]], out_dir: Path = FIGURES_DIR):
    """Plots bar chart comparison across Accuracy, Precision, Recall, and F1."""
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(models_results.keys())
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    accs = [models_results[m]["metrics"]["accuracy"] for m in models]
    precs = [models_results[m]["metrics"]["precision"] for m in models]
    recs = [models_results[m]["metrics"]["recall"] for m in models]
    f1s = [models_results[m]["metrics"]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, accs, width, label="Accuracy", color="#3182ce")
    ax.bar(x - 0.5 * width, precs, width, label="Precision", color="#38a169")
    ax.bar(x + 0.5 * width, recs, width, label="Recall", color="#dd6b20")
    ax.bar(x + 1.5 * width, f1s, width, label="F1-Score", color="#805ad5")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("TruthLens AI — Baseline & Multimodal Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="medium")
    ax.set_ylim([0.0, 1.15])
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Annotate values on top of bars
    for i in range(len(models)):
        ax.text(i - 1.5 * width, accs[i] + 0.02, f"{accs[i]:.2f}", ha="center", fontsize=8)
        ax.text(i - 0.5 * width, precs[i] + 0.02, f"{precs[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.5 * width, recs[i] + 0.02, f"{recs[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + 1.5 * width, f1s[i] + 0.02, f"{f1s[i]:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    out_file = out_dir / "metrics_comparison.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved metrics comparison bar chart to: {out_file}")
