"""Classification metrics (Section 4 reporting).

The paper reports, for the 3-class quality task:
    - macro-F1 (primary metric; early-stopping monitor)
    - accuracy (Acc)
    - balanced accuracy (BA)
    - Matthews correlation coefficient (MCC)
    - per-class precision / recall / F1

All metrics here wrap scikit-learn and return plain Python floats / dicts
so they serialize cleanly to JSON result files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from src.data.labels import CLASS_NAMES


@dataclass
class ClassificationMetrics:
    """Container for the metrics reported in Tables 6 and 8."""

    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    mcc: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    confusion: list[list[int]]  # row = true, col = pred

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "mcc": self.mcc,
            "per_class_f1": self.per_class_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "confusion": self.confusion,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int = 3,
) -> ClassificationMetrics:
    """Compute the full metric suite for a set of predictions.

    Parameters
    ----------
    y_true, y_pred
        Integer label arrays of equal length.
    n_classes
        Number of classes (paper: 3).
    """
    labels = list(range(n_classes))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    ba = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    names = list(CLASS_NAMES[:n_classes])
    if len(names) < n_classes:
        names.extend(f"Class{c}" for c in range(len(names), n_classes))
    per_class_f1 = {names[c]: float(f1[c]) for c in range(n_classes)}
    per_class_precision = {names[c]: float(prec[c]) for c in range(n_classes)}
    per_class_recall = {names[c]: float(rec[c]) for c in range(n_classes)}

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return ClassificationMetrics(
        macro_f1=macro_f1,
        accuracy=acc,
        balanced_accuracy=ba,
        mcc=mcc,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        confusion=cm,
    )


def summarize_cv(fold_metrics: list[ClassificationMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate per-fold metrics into mean +/- std (Table 6 format).

    Returns
    -------
    dict
        ``{"macro_f1": {"mean": ..., "std": ...}, "accuracy": {...}, ...}``
    """
    def agg(values: list[float]) -> dict[str, float]:
        arr = np.array(values, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    return {
        "macro_f1": agg([m.macro_f1 for m in fold_metrics]),
        "accuracy": agg([m.accuracy for m in fold_metrics]),
        "balanced_accuracy": agg([m.balanced_accuracy for m in fold_metrics]),
        "mcc": agg([m.mcc for m in fold_metrics]),
    }
