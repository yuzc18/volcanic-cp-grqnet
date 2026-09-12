"""Statistical utilities used by the finalized-manuscript analyses.

The functions in this module operate on *existing fitted predictions*.  They do
not refit models inside bootstrap replicates, matching Section 3.6 of the
manuscript.  The moving-block implementation is well-stratified and preserves
the number of observations contributed by each well in every replicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta, t

from src.training.metrics import ClassificationMetrics, compute_metrics


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float

    def to_dict(self) -> dict[str, float]:
        return {"estimate": float(self.estimate), "lower": float(self.lower), "upper": float(self.upper)}


def clopper_pearson_interval(
    covered: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> Interval:
    """Exact two-sided Clopper-Pearson binomial reference interval."""
    k = int(covered)
    n = int(total)
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Require 0 <= covered <= total and total > 0.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1).")
    tail = (1.0 - confidence) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - tail, k + 1, n - k))
    return Interval(float(k / n), lo, hi)


def paired_t_interval(
    paired_differences: Iterable[float],
    *,
    confidence: float = 0.95,
) -> Interval:
    """t-based interval for the mean of paired per-well differences."""
    d = np.asarray(list(paired_differences), dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("At least two finite paired differences are required.")
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    crit = float(t.ppf(0.5 + confidence / 2.0, df=len(d) - 1))
    return Interval(mean, mean - crit * se, mean + crit * se)


def _well_blocks(depths: np.ndarray, block_length_m: float) -> list[np.ndarray]:
    """All moving depth-window blocks for one sorted well.

    A block starts at every observed sample and contains consecutive samples
    from that start whose depth is < start_depth + block_length_m.  This is a
    conventional distance-based moving-block reconstruction for irregularly
    spaced core-matched observations.  It is documented because the manuscript
    states the 2.0 m block length but does not further specify endpoint handling.
    """
    if block_length_m <= 0:
        raise ValueError("block_length_m must be positive.")
    d = np.asarray(depths, dtype=float)
    if len(d) == 0:
        return []
    if np.any(np.diff(d) < 0):
        raise ValueError("Depths must be sorted ascending within a well.")
    blocks: list[np.ndarray] = []
    for start in range(len(d)):
        stop = int(np.searchsorted(d, d[start] + block_length_m, side="left"))
        stop = max(stop, start + 1)
        blocks.append(np.arange(start, min(stop, len(d)), dtype=np.int64))
    return blocks


def moving_block_resample_positions(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    block_length_m: float = 2.0,
    well_col: str = "well_id",
    depth_col: str = "depth_m",
) -> np.ndarray:
    """Return paired row positions from a well-stratified moving-block bootstrap.

    Every well contributes exactly its original number of rows.  Blocks are
    sampled with replacement inside each well and concatenated until that
    well's original row count is reached; excess positions from the final block
    are truncated.  Returned values are *positional* indices into ``frame``.
    """
    required = {well_col, depth_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    all_positions: list[np.ndarray] = []
    work = frame.reset_index(drop=True)
    for _, g in work.groupby(well_col, sort=False):
        positions = g.index.to_numpy(dtype=np.int64)
        order = np.argsort(g[depth_col].to_numpy(dtype=float), kind="stable")
        sorted_positions = positions[order]
        depths = g.iloc[order][depth_col].to_numpy(dtype=float)
        blocks = _well_blocks(depths, block_length_m)
        chosen_local: list[int] = []
        while len(chosen_local) < len(g):
            block = blocks[int(rng.integers(0, len(blocks)))]
            chosen_local.extend(block.tolist())
        chosen_local = chosen_local[: len(g)]
        all_positions.append(sorted_positions[np.asarray(chosen_local, dtype=np.int64)])
    return np.concatenate(all_positions) if all_positions else np.empty(0, dtype=np.int64)


def block_bootstrap_interval(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    n_resamples: int,
    block_length_m: float = 2.0,
    seed: int = 20260827,
    confidence: float = 0.95,
) -> Interval:
    """Well-stratified moving-block bootstrap interval for an existing statistic."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    estimate = float(statistic(frame))
    rng = np.random.default_rng(seed)
    reps = np.empty(int(n_resamples), dtype=float)
    work = frame.reset_index(drop=True)
    for b in range(int(n_resamples)):
        idx = moving_block_resample_positions(
            work, rng=rng, block_length_m=block_length_m
        )
        reps[b] = float(statistic(work.iloc[idx].reset_index(drop=True)))
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(reps, [tail, 1.0 - tail])
    return Interval(estimate, float(lo), float(hi))


def paired_model_block_bootstrap_difference(
    frame: pd.DataFrame,
    *,
    true_col: str,
    pred_a_col: str,
    pred_b_col: str,
    metric: str = "macro_f1",
    n_resamples: int = 2000,
    block_length_m: float = 2.0,
    seed: int = 20260827,
    confidence: float = 0.95,
) -> Interval:
    """Paired sample-level sensitivity interval for model A minus model B."""

    def stat(df: pd.DataFrame) -> float:
        a = compute_metrics(df[true_col].to_numpy(int), df[pred_a_col].to_numpy(int))
        b = compute_metrics(df[true_col].to_numpy(int), df[pred_b_col].to_numpy(int))
        if not hasattr(a, metric):
            raise ValueError(f"Unknown ClassificationMetrics field: {metric}")
        return float(getattr(a, metric) - getattr(b, metric))

    return block_bootstrap_interval(
        frame,
        stat,
        n_resamples=n_resamples,
        block_length_m=block_length_m,
        seed=seed,
        confidence=confidence,
    )


def within_well_coverage_block_interval(
    frame: pd.DataFrame,
    *,
    covered_col: str = "cp_covered",
    n_resamples: int = 1000,
    block_length_m: float = 2.0,
    seed: int = 20260827,
    confidence: float = 0.95,
) -> Interval:
    """Dependence-aware moving-block interval for empirical coverage."""
    return block_bootstrap_interval(
        frame,
        lambda df: float(df[covered_col].astype(bool).mean()),
        n_resamples=n_resamples,
        block_length_m=block_length_m,
        seed=seed,
        confidence=confidence,
    )


def greedy_thin_positions(
    frame: pd.DataFrame,
    *,
    minimum_spacing_m: float,
    starting_choice: int,
    well_col: str = "well_id",
    depth_col: str = "depth_m",
) -> np.ndarray:
    """Greedy within-well thinning from the first or second sample.

    ``starting_choice`` is 0 for the first sample and 1 for the second sample,
    exactly matching the two starting choices described in Section 4.4.
    """
    if starting_choice not in {0, 1}:
        raise ValueError("starting_choice must be 0 (first) or 1 (second).")
    if minimum_spacing_m <= 0:
        raise ValueError("minimum_spacing_m must be positive.")
    work = frame.reset_index(drop=True)
    selected: list[int] = []
    for _, g in work.groupby(well_col, sort=False):
        order = np.argsort(g[depth_col].to_numpy(float), kind="stable")
        positions = g.index.to_numpy(dtype=np.int64)[order]
        depths = g.iloc[order][depth_col].to_numpy(float)
        if len(positions) <= starting_choice:
            continue
        j = starting_choice
        selected.append(int(positions[j]))
        last_depth = float(depths[j])
        for k in range(j + 1, len(positions)):
            if float(depths[k]) - last_depth >= minimum_spacing_m - 1e-12:
                selected.append(int(positions[k]))
                last_depth = float(depths[k])
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class ThinningResult:
    minimum_spacing_m: float
    start0_n: int
    start1_n: int
    mean_macro_f1: float
    mean_accuracy: float
    mean_coverage: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "minimum_spacing_m": float(self.minimum_spacing_m),
            "start0_n": int(self.start0_n),
            "start1_n": int(self.start1_n),
            "mean_macro_f1": float(self.mean_macro_f1),
            "mean_accuracy": float(self.mean_accuracy),
            "mean_coverage": None if self.mean_coverage is None else float(self.mean_coverage),
        }


def thinning_analysis(
    frame: pd.DataFrame,
    *,
    spacings_m: Iterable[float] = (0.5, 1.0),
    true_col: str = "quality_class_id",
    pred_col: str = "predicted_class_id",
    covered_col: str | None = "cp_covered",
) -> list[ThinningResult]:
    """Re-evaluate existing predictions on the two greedy thinned subsets."""
    out: list[ThinningResult] = []
    for spacing in spacings_m:
        metrics: list[ClassificationMetrics] = []
        coverages: list[float] = []
        ns: list[int] = []
        for start in (0, 1):
            pos = greedy_thin_positions(
                frame, minimum_spacing_m=float(spacing), starting_choice=start
            )
            sub = frame.iloc[pos]
            ns.append(len(sub))
            metrics.append(compute_metrics(sub[true_col].to_numpy(int), sub[pred_col].to_numpy(int)))
            if covered_col is not None and covered_col in sub.columns:
                coverages.append(float(sub[covered_col].astype(bool).mean()))
        out.append(
            ThinningResult(
                minimum_spacing_m=float(spacing),
                start0_n=ns[0],
                start1_n=ns[1],
                mean_macro_f1=float(np.mean([m.macro_f1 for m in metrics])),
                mean_accuracy=float(np.mean([m.accuracy for m in metrics])),
                mean_coverage=None if not coverages else float(np.mean(coverages)),
            )
        )
    return out


def average_coverage_error(nominal_coverage: np.ndarray, empirical_coverage: np.ndarray) -> float:
    nominal = np.asarray(nominal_coverage, dtype=float)
    empirical = np.asarray(empirical_coverage, dtype=float)
    if nominal.shape != empirical.shape or nominal.ndim != 1:
        raise ValueError("nominal and empirical coverage arrays must be 1-D and aligned.")
    return float(np.mean(np.abs(empirical - nominal)))
