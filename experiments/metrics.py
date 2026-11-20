"""Evaluation metrics computation framework for TruthLens AI (Phase 4).

Computes comprehensive, scientifically reproducible classification metrics:
- Accuracy
- Precision (binary & macro)
- Recall (binary & macro)
- F1-Score (binary & macro)
- ROC-AUC
- PR-AUC (Precision-Recall Area Under Curve)
- Confusion Matrix (TP, FP, FN, TN)
- Class-wise metrics (Precision, Recall, F1, Support for REAL & FAKE)
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)


def compute_comprehensive_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics.

    Does not fabricate or smooth numbers; uses actual scikit-learn metrics.

    Args:
        y_true: Ground truth binary labels (0 = REAL, 1 = FAKE)
        y_pred: Model predicted binary labels (0 = REAL, 1 = FAKE)
        y_prob: Predicted probabilities for positive class (FAKE, class 1)

    Returns:
        Dict containing all required Phase 4 metrics.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    if len(y_true) == 0:
        return {
            "sample_count": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "pr_auc": None,
            "confusion_matrix": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
            "class_metrics": {},
        }

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    # cm layout:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Class-wise metrics
    # Class 0: REAL
    real_prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    real_rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    real_f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    real_supp = int(np.sum(y_true_arr == 0))

    # Class 1: FAKE
    fake_prec = prec
    fake_rec = rec
    fake_f1 = f1
    fake_supp = int(np.sum(y_true_arr == 1))

    # ROC-AUC and PR-AUC
    roc_auc_val: Optional[float] = None
    pr_auc_val: Optional[float] = None

    if y_prob is not None and len(y_prob) == len(y_true):
        y_prob_arr = np.array(y_prob, dtype=float)
        # Check if both classes are present in y_true
        if len(np.unique(y_true_arr)) > 1:
            try:
                roc_auc_val = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4)
            except Exception:
                roc_auc_val = None

            try:
                precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)
                pr_auc_val = round(float(auc(recall_curve, precision_curve)), 4)
            except Exception:
                pr_auc_val = None

    return {
        "sample_count": len(y_true),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "class_metrics": {
            "REAL (Class 0)": {
                "precision": round(real_prec, 4),
                "recall": round(real_rec, 4),
                "f1": round(real_f1, 4),
                "support": real_supp,
            },
            "FAKE (Class 1)": {
                "precision": round(fake_prec, 4),
                "recall": round(fake_rec, 4),
                "f1": round(fake_f1, 4),
                "support": fake_supp,
            },
        },
        "support": {
            "real": real_supp,
            "fake": fake_supp,
            "total": len(y_true),
        },
    }
