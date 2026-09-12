"""Machine-readable summary tables derived from supplied study-format data/predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import f1_score

from src.data.labels import compute_fzi


def petrophysical_summary(labeled_core: pd.DataFrame) -> pd.DataFrame:
    """Table-4 style medians/IQRs from recorded core values and active labels."""
    f = labeled_core.copy()
    f["fzi_um"] = compute_fzi(f["porosity_pct"], f["permeability_mD"])
    names = {0: "High", 1: "Medium", 2: "Low"}
    rows = []
    for c in sorted(f["quality_class_id"].unique()):
        g = f.loc[f["quality_class_id"] == c]
        row = {"quality_class_id": int(c), "quality_class": names.get(int(c), f"Class{c}"), "n": len(g)}
        for col, prefix in (("porosity_pct", "porosity_pct"), ("permeability_mD", "permeability_mD"), ("fzi_um", "fzi_um")):
            q1, med, q3 = np.quantile(g[col].to_numpy(float), [0.25, 0.5, 0.75])
            row[f"{prefix}_median"] = float(med)
            row[f"{prefix}_q1"] = float(q1)
            row[f"{prefix}_q3"] = float(q3)
        rows.append(row)
    return pd.DataFrame(rows)


def classification_summary(pred: pd.DataFrame, dataset_name: str) -> dict:
    y = pred["quality_class_id"].to_numpy(int)
    yp = pred["predicted_class_id"].to_numpy(int)
    labels = sorted(np.unique(np.concatenate([y, yp])))
    f1s = f1_score(y, yp, labels=labels, average=None, zero_division=0)
    out = {"dataset": dataset_name, "n": len(pred), "macro_f1": float(f1_score(y, yp, average="macro", zero_division=0))}
    names = {0: "High", 1: "Medium", 2: "Low"}
    for c, v in zip(labels, f1s, strict=True):
        out[f"{names.get(int(c), f'class_{c}')}_f1"] = float(v)
    return out


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def coverage_summary(pred: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if "cp_covered" not in pred:
        raise ValueError("prediction frame must contain cp_covered")
    names = {0: "High", 1: "Medium", 2: "Low"}
    strata: list[tuple[str, pd.DataFrame]] = [("Marginal", pred)]
    for c in sorted(pred["quality_class_id"].unique()):
        strata.append((names.get(int(c), f"Class{c}"), pred.loc[pred["quality_class_id"] == c]))
    rows = []
    for name, g in strata:
        k = int(g["cp_covered"].astype(bool).sum())
        n = len(g)
        lo, hi = clopper_pearson(k, n)
        rows.append({
            "dataset": dataset_name, "stratum": name, "covered": k, "total": n,
            "empirical_coverage": k / n if n else float("nan"), "cp95_low": lo, "cp95_high": hi,
        })
    return pd.DataFrame(rows)
