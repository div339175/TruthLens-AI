"""Confusion matrix generator and formatter."""

from typing import List, Dict, Any
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Generates confusion matrix and formatted counts."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "matrix": cm.tolist(),
        "true_negatives": int(tn),  # Real classified as Real
        "false_positives": int(fp),  # Real classified as Fake
        "false_negatives": int(fn),  # Fake classified as Real
        "true_positives": int(tp),  # Fake classified as Fake
        "labels": ["REAL (0)", "FAKE (1)"],
    }
