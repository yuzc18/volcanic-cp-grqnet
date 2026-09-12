"""Matplotlib figure builders corresponding to manuscript Figures 6--12.

The builders contain no manuscript result constants beyond fixed method
settings such as the frozen FZI boundaries supplied by the caller.  All
annotations are calculated from the input frames so synthetic/demo runs are
clearly allowed to differ from the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.analysis.conformal_analysis import (
    CoverageCurve,
    VerificationCurve,
    accuracy_by_set_size,
    fig11_groups,
    set_size_by_class,
)
from src.data.io import LWD_COLUMNS
from src.data.labels import CLASS_NAMES, FZIBoundaries, RQI_CONSTANT


def _save(fig, output: str | Path | None):
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def figure6_petrophysics(
    frame: pd.DataFrame,
    boundaries: FZIBoundaries,
    *,
    output: str | Path | None = None,
):
    """Figure 6: porosity-permeability crossplot and FZI distributions."""
    required = {"porosity_pct", "permeability_mD", "FZI_um", "quality_class_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Figure 6 frame missing columns: {sorted(missing)}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for cls in range(3):
        g = frame.loc[frame["quality_class_id"] == cls]
        ax.scatter(g["porosity_pct"], g["permeability_mD"], s=12, alpha=0.65, label=CLASS_NAMES[cls])
    phi_pct = np.linspace(max(0.2, frame["porosity_pct"].min() * 0.8), frame["porosity_pct"].max() * 1.1, 250)
    phi = phi_pct / 100.0
    for b, label in [(boundaries.medium_low, f"FZI={boundaries.medium_low:.2f}"), (boundaries.high_medium, f"FZI={boundaries.high_medium:.2f}")]:
        k = phi * (b * phi / (RQI_CONSTANT * (1.0 - phi))) ** 2
        ax.plot(phi_pct, k, linestyle="--", linewidth=1.3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Porosity (%)")
    ax.set_ylabel("Permeability (mD)")
    ax.set_title("(a) Porosity-permeability crossplot")
    ax.legend(fontsize=8)

    ax = axes[1]
    values = [frame.loc[frame["quality_class_id"] == c, "FZI_um"].to_numpy(float) for c in range(3)]
    ax.boxplot(values, labels=CLASS_NAMES, whis=(0, 100), showfliers=False)
    ax.axhline(boundaries.medium_low, linestyle="--", linewidth=1.2)
    ax.axhline(boundaries.high_medium, linestyle="--", linewidth=1.2)
    ax.set_ylabel("FZI (μm)")
    ax.set_title("(b) FZI distributions")
    fig.tight_layout()
    return _save(fig, output)


def figure7_confusion_matrices(
    predictions: pd.DataFrame,
    *,
    model_order: Sequence[str] = ("mlp", "transformer", "lightgbm", "tabnet", "catboost", "grqnet"),
    output: str | Path | None = None,
):
    """Figure 7: pooled row-normalized confusion matrices for six models."""
    required = {"model", "quality_class_id", "predicted_class_id"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Figure 7 predictions missing columns: {sorted(missing)}")
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.0))
    for ax, name in zip(axes.ravel(), model_order):
        g = predictions.loc[predictions["model"] == name]
        if g.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Missing: {name}", ha="center", va="center")
            continue
        cm = np.zeros((3, 3), dtype=float)
        y = g["quality_class_id"].to_numpy(int)
        p = g["predicted_class_id"].to_numpy(int)
        for i in range(3):
            for j in range(3):
                cm[i, j] = np.sum((y == i) & (p == j))
        row = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0) * 100.0
        im = ax.imshow(pct, vmin=0, vmax=100, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(3), CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Row percentage (%)")
    return _save(fig, output)


def _display_curve_points(curve: CoverageCurve, n: int = 9) -> np.ndarray:
    # Display-only selection: evenly spaced indices across the *already fixed*
    # 30-point analysis grid. This does not change ACE or any conformal result.
    return np.unique(np.rint(np.linspace(0, len(curve.alpha) - 1, n)).astype(int))


def figure8_conformal_behavior(
    curve: CoverageCurve,
    primary_oof: pd.DataFrame,
    *,
    output: str | Path | None = None,
):
    """Figure 8: calibration curve, set-size distribution, and accuracy by size."""
    dist = set_size_by_class(primary_oof)
    acc = accuracy_by_set_size(primary_oof)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))

    ax = axes[0]
    idx = _display_curve_points(curve, 9)
    x = curve.nominal_coverage[idx]
    y = curve.empirical_coverage[idx]
    order = np.argsort(x)
    ax.plot(x[order], y[order], marker="o", linewidth=1.5)
    # Manuscript Fig. 8(a) displays nine levels sampled from the 30-point
    # alpha grid and additionally shows the trivial nominal-coverage limit
    # (1.00, 1.00) as a reference point only.  It is not part of ACE.
    ax.plot([1.0], [1.0], marker="o", markersize=5, markerfacecolor="none",
            markeredgewidth=1.0, linestyle="None")
    lo = float(min(curve.nominal_coverage.min(), curve.empirical_coverage.min()))
    hi = 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Expected coverage (1−α)")
    ax.set_ylabel("Observed coverage")
    ax.set_title("(a) Calibration curve")
    ax.text(0.03, 0.95, f"ACE = {curve.ace:.4f}", transform=ax.transAxes, va="top")

    ax = axes[1]
    bottom = np.zeros(3)
    class_ids = np.arange(3)
    for size in (1, 2, 3):
        vals = np.array([
            float(dist.loc[(dist.quality_class_id == c) & (dist.set_size == size), "percentage"].iloc[0])
            for c in class_ids
        ])
        ax.bar(class_ids, vals, bottom=bottom, label=f"|C|={size}")
        bottom += vals
    for c in class_ids:
        mean_size = float(dist.loc[dist.quality_class_id == c, "mean_set_size"].iloc[0])
        n_cls = int(dist.loc[dist.quality_class_id == c, "n_class"].iloc[0])
        ax.text(c, 102, f"m={mean_size:.2f}\n(n={n_cls})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(class_ids, CLASS_NAMES)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(b) CP set-size distribution")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.bar(acc["set_size"].astype(str), 100 * acc["accuracy"])
    for i, row in acc.reset_index(drop=True).iterrows():
        ax.text(i, 100 * row["accuracy"] + 1.5, f"{100*row['accuracy']:.1f}%", ha="center", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Prediction-set size")
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title("(c) Accuracy by prediction-set size")
    fig.tight_layout()
    return _save(fig, output)


def figure9_blind_profile(
    frame: pd.DataFrame,
    *,
    q_hat: float,
    well_id: str = "CS9",
    output: str | Path | None = None,
):
    """Figure 9: composite blind-well profile at core-matched depths."""
    required = {"well_id", "depth_m", *LWD_COLUMNS, "quality_class_id", "predicted_class_id", "prob_High", "prob_Medium", "prob_Low", "cp_set_size"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Figure 9 frame missing columns: {sorted(missing)}")
    g = frame.loc[frame["well_id"].astype(str) == str(well_id)].sort_values("depth_m")
    if g.empty:
        raise ValueError(f"No rows found for blind well {well_id}.")
    depth = g["depth_m"].to_numpy(float)
    tracks = ["RLA5", "GR", "CNL", "AC", "DEN"]
    fig, axes = plt.subplots(1, 9, figsize=(14.5, 7.4), sharey=True)
    for ax, col in zip(axes[:5], tracks):
        ax.plot(g[col], depth, linewidth=0.8)
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("Depth (m)")

    axes[5].scatter(g["quality_class_id"], depth, s=12)
    axes[5].set_xticks(range(3), ["H", "M", "L"])
    axes[5].set_title("True", fontsize=9)
    mis = g["predicted_class_id"].to_numpy(int) != g["quality_class_id"].to_numpy(int)
    axes[6].scatter(g["predicted_class_id"], depth, s=12)
    axes[6].scatter(g.loc[mis, "predicted_class_id"], depth[mis], marker="^", s=24, color="red")
    axes[6].set_xticks(range(3), ["H", "M", "L"])
    axes[6].set_title("Pred", fontsize=9)

    axes[7].plot(g["prob_High"], depth, label="High")
    axes[7].plot(g["prob_Medium"], depth, label="Medium")
    axes[7].plot(g["prob_Low"], depth, label="Low")
    axes[7].set_xlim(0, 1)
    axes[7].set_title("Class probability", fontsize=9)
    axes[7].legend(fontsize=6)

    max_prob = g[["prob_High", "prob_Medium", "prob_Low"]].max(axis=1).to_numpy(float)
    singleton = g["cp_set_size"].to_numpy(int) == 1
    axes[8].scatter(max_prob[singleton], depth[singleton], s=12, color="gray", label="|C|=1")
    axes[8].scatter(max_prob[~singleton], depth[~singleton], s=12, color="purple", label="|C|≥2")
    axes[8].axvline(float(q_hat), linestyle="--", linewidth=1.0)
    axes[8].set_xlim(0, 1)
    axes[8].set_title("Confidence", fontsize=9)
    axes[8].legend(fontsize=6)
    axes[8].text(float(q_hat), depth.min(), f"q̂={q_hat:.3f}", fontsize=7, va="bottom")

    for ax in axes:
        ax.invert_yaxis()
        ax.grid(alpha=0.15)
    fig.tight_layout()
    return _save(fig, output)


def figure10_interpretability(
    feature_importance: pd.DataFrame,
    group_importance: pd.DataFrame,
    oof_frame: pd.DataFrame,
    *,
    top_n: int = 10,
    output: str | Path | None = None,
):
    """Figure 10: single-feature SHAP, group SHAP/weights, weight distributions."""
    fig = plt.figure(figsize=(10.6, 7.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    groups = ["A", "B", "C", "D"]
    top = feature_importance.head(top_n).iloc[::-1]
    group_colors = {"A": "#9ecae1", "B": "#fdd0a2", "C": "#a1d99b", "D": "#f7b6d2"}
    ax1.barh(top["feature"], top["mean_abs_shap"], color=[group_colors.get(g, "gray") for g in top["group"]])
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor=group_colors[g], label=f"Group {g}") for g in groups], fontsize=6)
    ax1.set_xlabel("Mean |SHAP|")
    ax1.set_title("(a) Single-feature SHAP")

    shap_vals = np.array([
        float(group_importance.loc[group_importance.group == g, "mean_feature_abs_shap"].iloc[0])
        for g in groups
    ])
    weight_vals = np.array([float(oof_frame[f"alpha_{g}"].mean()) for g in groups])
    shap_norm = shap_vals / shap_vals.max() if shap_vals.max() > 0 else shap_vals
    weight_norm = weight_vals / weight_vals.max() if weight_vals.max() > 0 else weight_vals
    ax2.plot(groups, shap_norm, marker="o", label="Mean feature-level SHAP")
    ax2.plot(groups, weight_norm, marker="s", linestyle="--", label="Group weight α")
    ax2.set_ylabel("Normalized value")
    ax2.set_title("(b) Group SHAP and weights")
    ax2.legend(fontsize=7)

    positions = []
    data = []
    labels = []
    pos = 1
    for group in groups:
        for cls in range(3):
            vals = oof_frame.loc[oof_frame["quality_class_id"] == cls, f"alpha_{group}"].to_numpy(float)
            data.append(vals)
            positions.append(pos)
            labels.append(f"{group}-{CLASS_NAMES[cls][0]}")
            pos += 1
        pos += 1
    bp = ax3.boxplot(data, positions=positions, widths=0.65, showfliers=False, patch_artist=True)
    class_colors = ["#6baed6", "#fdae6b", "#bdbdbd"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(class_colors[i % 3])
    ax3.legend(handles=[Patch(facecolor=class_colors[c], label=CLASS_NAMES[c]) for c in range(3)], fontsize=7)
    ax3.set_xticks(positions, labels, rotation=45, ha="right")
    ax3.set_ylabel("Group weight α")
    ax3.set_title("(c) Group-weight distributions by reservoir quality class")
    fig.tight_layout()
    return _save(fig, output)


def _kde_line(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return None
    lo, hi = np.quantile(values, [0.005, 0.995])
    pad = max((hi - lo) * 0.15, 1e-3)
    x = np.linspace(max(1e-6, lo - pad), hi + pad, 300)
    kde = gaussian_kde(values)
    y = kde(x)
    if y.max() > 0:
        y = y / y.max()
    return x, y


def figure11_fzi_prediction_sets(
    oof_frame: pd.DataFrame,
    boundaries: FZIBoundaries,
    *,
    output: str | Path | None = None,
):
    """Figure 11: independently normalized FZI KDEs by set composition/size."""
    compositions, sizes = fig11_groups(oof_frame)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for label in ("Medium+Low", "High+Medium", "High+Low"):
        res = _kde_line(compositions[label])
        if res is not None:
            axes[0].plot(*res, label=f"{label} (n={len(compositions[label])})")
    axes[0].set_title("(a) Size-two set composition")
    axes[0].set_xlabel("FZI (μm)")
    axes[0].set_ylabel("Normalized KDE")
    axes[0].legend(fontsize=7)
    for label in ("size1", "size2", "size3"):
        res = _kde_line(sizes[label])
        if res is not None:
            axes[1].plot(*res, label=f"|C|={label[-1]} (n={len(sizes[label])})")
    axes[1].set_title("(b) Prediction-set size")
    axes[1].set_xlabel("FZI (μm)")
    axes[1].set_ylabel("Normalized KDE")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.axvline(boundaries.medium_low, linestyle="--", linewidth=1.0)
        ax.axvline(boundaries.high_medium, linestyle="--", linewidth=1.0)
    fig.tight_layout()
    return _save(fig, output)


def figure12_verification_budget(
    curve: VerificationCurve,
    *,
    output: str | Path | None = None,
):
    """Figure 12: misclassification recovery against verification budget."""
    x = 100.0 * curve.budget_fraction
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(x, 100 * curve.cp_guided, color="tab:blue", label="CP-guided")
    ax.plot(x, 100 * curve.softmax, color="tab:orange", label="Softmax")
    ax.plot(x, 100 * curve.uniform_random, color="gray", linestyle="--", label="Uniform random")
    if curve.cp_lower is not None:
        ax.fill_between(x, 100 * curve.cp_lower, 100 * curve.cp_upper, color="tab:blue", alpha=0.15)
        ax.fill_between(x, 100 * curve.softmax_lower, 100 * curve.softmax_upper, color="tab:orange", alpha=0.15)
        ax.fill_between(x, 100 * curve.uniform_lower, 100 * curve.uniform_upper, color="gray", alpha=0.15)
    idx = int(np.argmin(np.abs(curve.budget_fraction - 0.10)))
    for y, label in [
        (curve.cp_guided[idx], "CP"),
        (curve.softmax[idx], "Softmax"),
        (curve.uniform_random[idx], "Uniform"),
    ]:
        ax.scatter([x[idx]], [100 * y], facecolors="white", edgecolors="black", zorder=5)
        ax.text(x[idx] + 0.5, 100 * y + 1.0, f"{100*y:.1f}%", fontsize=8)
    ax.set_xlabel("Verification budget (%)")
    ax.set_ylabel("Misclassified samples recovered (%)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _save(fig, output)
