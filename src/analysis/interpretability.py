"""GRQ-Net interpretability analyses for Figure 10.

The manuscript-facing SHAP protocol is fold-wise DeepExplainer on model logits,
with 100 background rows sampled from that fold's training subset.  Feature
importance is mean absolute SHAP over validation samples and the three class
outputs; group-level SHAP is the arithmetic mean of feature-level values within
each UFT group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.features.uft import UFT_COLUMNS, UFT_LAYOUT

GROUP_NAMES = ["A", "B", "C", "D"]
GROUP_SLICES = [UFT_LAYOUT.slice_a, UFT_LAYOUT.slice_b, UFT_LAYOUT.slice_c, UFT_LAYOUT.slice_d]


@dataclass(frozen=True)
class ShapSummary:
    feature_importance: pd.DataFrame
    group_importance: pd.DataFrame
    group_weight_summary: pd.DataFrame


def _normalise_shap_array(values, n_samples: int, n_features: int, n_outputs: int) -> np.ndarray:
    """Return SHAP values as (sample, feature, output) across SHAP versions."""
    if isinstance(values, list):
        # Historical SHAP: one (sample, feature) array per output.
        arr = np.stack([np.asarray(v) for v in values], axis=-1)
    else:
        arr = np.asarray(values)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected DeepExplainer SHAP shape {arr.shape!r}.")
    if arr.shape == (n_samples, n_features, n_outputs):
        return arr
    if arr.shape == (n_outputs, n_samples, n_features):
        return np.moveaxis(arr, 0, -1)
    if arr.shape == (n_samples, n_outputs, n_features):
        return np.moveaxis(arr, 1, -1)
    raise ValueError(
        f"Cannot map DeepExplainer output shape {arr.shape!r} to "
        f"({n_samples}, {n_features}, {n_outputs})."
    )


def compute_foldwise_shap(
    fold_runs,
    *,
    background_size: int = 100,
    seed: int = 20260827,
    max_eval_rows_per_fold: int | None = None,
) -> ShapSummary:
    """Compute Figure-10 SHAP and group-weight summaries from trained outer folds."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise RuntimeError("SHAP is required; install requirements.txt.") from exc

    rng = np.random.default_rng(seed)
    weighted_feature_sum = np.zeros(len(UFT_COLUMNS), dtype=float)
    total_eval = 0
    weight_frames: list[pd.DataFrame] = []

    for run in fold_runs:
        X_train = np.asarray(run.prepared.X_outer_train, dtype=np.float32)
        X_eval = np.asarray(run.prepared.X_outer_val, dtype=np.float32)
        if max_eval_rows_per_fold is not None:
            X_eval = X_eval[: int(max_eval_rows_per_fold)]
        n_bg = min(int(background_size), len(X_train))
        if n_bg <= 0 or len(X_eval) <= 0:
            raise ValueError("SHAP requires non-empty train and validation matrices.")
        bg_idx = rng.choice(len(X_train), size=n_bg, replace=False)
        background = torch.from_numpy(X_train[bg_idx])
        eval_tensor = torch.from_numpy(X_eval)

        model = run.train_result.model
        model.eval()
        # GRQNet.forward returns raw logits by default, which is the manuscript
        # target for DeepExplainer.
        explainer = shap.DeepExplainer(model, background)
        values = explainer.shap_values(eval_tensor, check_additivity=False)
        arr = _normalise_shap_array(values, len(X_eval), len(UFT_COLUMNS), 3)
        fold_imp = np.abs(arr).mean(axis=(0, 2))
        weighted_feature_sum += fold_imp * len(X_eval)
        total_eval += len(X_eval)

        wf = run.oof_frame.iloc[: len(X_eval)][
            ["sample_id", "well_id", "quality_class_id", "alpha_A", "alpha_B", "alpha_C", "alpha_D"]
        ].copy()
        weight_frames.append(wf)

    feature_imp = weighted_feature_sum / float(total_eval)
    feature_groups = (
        ["A"] * UFT_LAYOUT.dim_a
        + ["B"] * UFT_LAYOUT.dim_b
        + ["C"] * UFT_LAYOUT.dim_c
        + ["D"] * UFT_LAYOUT.dim_d
    )
    feature_frame = pd.DataFrame(
        {"feature": UFT_COLUMNS, "group": feature_groups, "mean_abs_shap": feature_imp}
    ).sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)

    group_rows = []
    for name, sl in zip(GROUP_NAMES, GROUP_SLICES):
        group_rows.append(
            {"group": name, "mean_feature_abs_shap": float(feature_imp[sl].mean())}
        )
    group_frame = pd.DataFrame(group_rows)

    weights = pd.concat(weight_frames, ignore_index=True)
    long = weights.melt(
        id_vars=["sample_id", "well_id", "quality_class_id"],
        value_vars=["alpha_A", "alpha_B", "alpha_C", "alpha_D"],
        var_name="group",
        value_name="group_weight",
    )
    long["group"] = long["group"].str.replace("alpha_", "", regex=False)
    weight_summary = (
        long.groupby(["quality_class_id", "group"], as_index=False)["group_weight"]
        .agg(["count", "median", "mean", "std"])
        .reset_index()
    )
    return ShapSummary(feature_frame, group_frame, weight_summary)
