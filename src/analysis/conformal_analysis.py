"""Post-fit conformal analyses for Figures 8, 11 and 12.

All functions consume existing out-of-fold probabilities/prediction sets.  They
never alter or refit the predictive models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.analysis.statistics import average_coverage_error, moving_block_resample_positions
from src.uncertainty.conformal import (
    aps_nonconformity_scores,
    build_prediction_sets_detailed,
    compute_threshold,
)
from src.uncertainty.workflow import PROB_COLS


@dataclass(frozen=True)
class CoverageCurve:
    alpha: np.ndarray
    nominal_coverage: np.ndarray
    empirical_coverage: np.ndarray
    ace: float

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "alpha": self.alpha,
                "nominal_coverage": self.nominal_coverage,
                "empirical_coverage": self.empirical_coverage,
            }
        )


def coverage_curve_from_folds(
    fold_runs,
    *,
    alpha_grid: Iterable[float] = tuple(np.arange(0.01, 0.301, 0.01)),
    seed: int = 20260827,
) -> CoverageCurve:
    """Recompute q-hat across alpha while holding fitted probabilities fixed.

    Randomized APS uniforms are drawn once from a single ``default_rng`` stream
    in manuscript fold order and then reused across the entire alpha grid.  Thus
    only the conformal threshold changes with alpha.
    """
    alphas = np.asarray(list(alpha_grid), dtype=float)
    if len(alphas) == 0 or np.any((alphas <= 0) | (alphas >= 1)):
        raise ValueError("alpha_grid must contain values in (0, 1).")

    rng = np.random.default_rng(seed)
    cached: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for run in fold_runs:
        p_cal = run.calibration_frame[PROB_COLS].to_numpy(float)
        y_cal = run.calibration_frame["quality_class_id"].to_numpy(int)
        p_eval = run.oof_frame[PROB_COLS].to_numpy(float)
        y_eval = run.oof_frame["quality_class_id"].to_numpy(int)
        u_cal = rng.uniform(0.0, 1.0, size=len(y_cal))
        scores = aps_nonconformity_scores(p_cal, y_cal, uniforms=u_cal, randomized=True)
        u_eval = rng.uniform(0.0, 1.0, size=len(y_eval))
        cached.append((scores, p_eval, y_eval, u_eval))

    empirical = np.empty(len(alphas), dtype=float)
    for j, alpha in enumerate(alphas):
        covered_all: list[np.ndarray] = []
        for scores, p_eval, y_eval, u_eval in cached:
            # The 30-point grid starts at alpha=0.01, where ceil((n+1)(1-alpha))
            # can exceed a small calibration set.  The finite-sample threshold
            # is then infinite, i.e. the trivial full-label set, which is the
            # correct conservative value for a coverage curve.  The manuscript
            # designs never reach this branch: alpha=0.01 needs n>=99 and the
            # primary calibration set has n=178.
            q_hat = compute_threshold(
                scores, float(alpha), on_insufficient_calibration="full_set"
            )
            details = build_prediction_sets_detailed(
                p_eval,
                q_hat,
                uniforms=u_eval,
                randomized=True,
                always_retain_top=True,
            )
            covered_all.append(
                np.asarray([int(y_eval[i]) in details.sets[i] for i in range(len(y_eval))], dtype=bool)
            )
        empirical[j] = float(np.concatenate(covered_all).mean())

    nominal = 1.0 - alphas
    return CoverageCurve(
        alpha=alphas,
        nominal_coverage=nominal,
        empirical_coverage=empirical,
        ace=average_coverage_error(nominal, empirical),
    )


def set_size_by_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Figure-8(b) percentages and mean set size by true quality class."""
    rows = []
    for cls, g in frame.groupby("quality_class_id", sort=True):
        total = len(g)
        mean_size = float(g["cp_set_size"].mean())
        for size in (1, 2, 3):
            rows.append(
                {
                    "quality_class_id": int(cls),
                    "set_size": size,
                    "count": int((g["cp_set_size"] == size).sum()),
                    "percentage": float(100.0 * (g["cp_set_size"] == size).mean()),
                    "mean_set_size": mean_size,
                    "n_class": total,
                }
            )
    return pd.DataFrame(rows)


def accuracy_by_set_size(frame: pd.DataFrame) -> pd.DataFrame:
    """Figure-8(c) point-classification accuracy stratified by set size."""
    rows = []
    for size, g in frame.groupby("cp_set_size", sort=True):
        correct = g["predicted_class_id"].to_numpy(int) == g["quality_class_id"].to_numpy(int)
        rows.append(
            {
                "set_size": int(size),
                "n": int(len(g)),
                "n_correct": int(correct.sum()),
                "accuracy": float(correct.mean()),
            }
        )
    return pd.DataFrame(rows)


def decode_cp_set(value: str) -> tuple[int, ...]:
    value = str(value).strip()
    if not value:
        return ()
    return tuple(sorted(int(x) for x in value.split("|") if x != ""))


def fig11_groups(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return FZI vectors grouped by size-two composition and by set size."""
    if "FZI_um" not in frame.columns:
        raise ValueError("Figure 11 requires FZI_um merged into the OOF frame.")
    comp = {"Medium+Low": [], "High+Medium": [], "High+Low": []}
    size_groups = {"size1": [], "size2": [], "size3": []}
    for _, row in frame.iterrows():
        fzi = float(row["FZI_um"])
        s = decode_cp_set(row["cp_set"])
        size_groups[f"size{len(s)}"].append(fzi)
        if len(s) == 2:
            if s == (1, 2):
                comp["Medium+Low"].append(fzi)
            elif s == (0, 1):
                comp["High+Medium"].append(fzi)
            elif s == (0, 2):
                comp["High+Low"].append(fzi)
    return (
        {k: np.asarray(v, dtype=float) for k, v in comp.items()},
        {k: np.asarray(v, dtype=float) for k, v in size_groups.items()},
    )


def _recovery_curve_for_order(errors: np.ndarray, order: np.ndarray, budget_counts: np.ndarray) -> np.ndarray:
    total_errors = int(errors.sum())
    if total_errors <= 0:
        return np.zeros(len(budget_counts), dtype=float)
    cum = np.cumsum(errors[order].astype(int))
    out = np.empty(len(budget_counts), dtype=float)
    for i, k in enumerate(budget_counts):
        if k <= 0:
            out[i] = 0.0
        else:
            out[i] = float(cum[min(int(k), len(cum)) - 1] / total_errors)
    return out


def _random_tie_order(values: np.ndarray, *, descending: bool, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values)
    keys = -values if descending else values
    jitter = rng.random(len(values))
    # lexsort uses last key as primary: numeric order first, random tie key second.
    return np.lexsort((jitter, keys))


@dataclass(frozen=True)
class VerificationCurve:
    budget_fraction: np.ndarray
    cp_guided: np.ndarray
    softmax: np.ndarray
    uniform_random: np.ndarray
    cp_lower: np.ndarray | None = None
    cp_upper: np.ndarray | None = None
    softmax_lower: np.ndarray | None = None
    softmax_upper: np.ndarray | None = None
    uniform_lower: np.ndarray | None = None
    uniform_upper: np.ndarray | None = None
    cp_minus_uniform_lower: np.ndarray | None = None
    cp_minus_uniform_upper: np.ndarray | None = None
    cp_minus_softmax_lower: np.ndarray | None = None
    cp_minus_softmax_upper: np.ndarray | None = None

    def as_frame(self) -> pd.DataFrame:
        data = {
            "budget_fraction": self.budget_fraction,
            "cp_guided": self.cp_guided,
            "softmax": self.softmax,
            "uniform_random": self.uniform_random,
        }
        for name in (
            "cp_lower", "cp_upper", "softmax_lower", "softmax_upper", "uniform_lower", "uniform_upper",
            "cp_minus_uniform_lower", "cp_minus_uniform_upper", "cp_minus_softmax_lower", "cp_minus_softmax_upper"
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return pd.DataFrame(data)


def verification_recovery_curves(
    frame: pd.DataFrame,
    *,
    budget_fraction: Iterable[float] = tuple(np.arange(0.05, 0.301, 0.01)),
    random_draws: int = 200,
    seed: int = 20260827,
) -> VerificationCurve:
    """Original-prediction curves used by Figure 12."""
    budgets = np.asarray(list(budget_fraction), dtype=float)
    if np.any((budgets <= 0) | (budgets > 1)):
        raise ValueError("budget fractions must lie in (0, 1].")
    n = len(frame)
    budget_counts = np.floor(budgets * n + 1e-12).astype(int)
    budget_counts = np.maximum(budget_counts, 1)
    errors = (frame["predicted_class_id"].to_numpy(int) != frame["quality_class_id"].to_numpy(int))
    max_prob = frame[PROB_COLS].to_numpy(float).max(axis=1)
    set_size = frame["cp_set_size"].to_numpy(int)
    rng = np.random.default_rng(seed)

    cp_reps = np.empty((random_draws, len(budgets)), dtype=float)
    uniform_reps = np.empty_like(cp_reps)
    for r in range(random_draws):
        cp_order = _random_tie_order(set_size, descending=True, rng=rng)
        uniform_order = rng.permutation(n)
        cp_reps[r] = _recovery_curve_for_order(errors, cp_order, budget_counts)
        uniform_reps[r] = _recovery_curve_for_order(errors, uniform_order, budget_counts)
    softmax_order = np.argsort(max_prob, kind="stable")
    softmax = _recovery_curve_for_order(errors, softmax_order, budget_counts)
    return VerificationCurve(
        budget_fraction=budgets,
        cp_guided=cp_reps.mean(axis=0),
        softmax=softmax,
        uniform_random=uniform_reps.mean(axis=0),
    )


def verification_block_bootstrap(
    frame: pd.DataFrame,
    *,
    budget_fraction: Iterable[float] = tuple(np.arange(0.05, 0.301, 0.01)),
    random_draws: int = 200,
    n_resamples: int = 1000,
    block_length_m: float = 2.0,
    seed: int = 20260827,
) -> VerificationCurve:
    """Figure-12 paired well-stratified moving-block bootstrap intervals.

    Each bootstrap replicate resamples existing predictions within wells, then
    rebuilds all rankings and redraws CP tie-breaking and uniform selection.
    """
    work = frame.reset_index(drop=True)
    point = verification_recovery_curves(
        work, budget_fraction=budget_fraction, random_draws=random_draws, seed=seed
    )
    budgets = point.budget_fraction
    rng = np.random.default_rng(seed)
    cp = np.empty((n_resamples, len(budgets)), dtype=float)
    sm = np.empty_like(cp)
    un = np.empty_like(cp)
    for b in range(n_resamples):
        idx = moving_block_resample_positions(work, rng=rng, block_length_m=block_length_m)
        resampled = work.iloc[idx].reset_index(drop=True)
        # Fresh seed drawn from the bootstrap RNG means tie-breaking and uniform
        # permutations are redrawn within every replicate while keeping the
        # three strategies paired to the same resampled rows.
        rep_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        # The manuscript line estimates average the random strategies over R=200,
        # but each bootstrap replicate redraws the randomized ranking once.
        # Therefore bootstrap uncertainty includes single-selection randomness
        # rather than averaging it away inside each replicate.
        curve = verification_recovery_curves(
            resampled,
            budget_fraction=budgets,
            random_draws=1,
            seed=rep_seed,
        )
        cp[b] = curve.cp_guided
        sm[b] = curve.softmax
        un[b] = curve.uniform_random
    qlo, qhi = 0.025, 0.975
    return VerificationCurve(
        budget_fraction=budgets,
        cp_guided=point.cp_guided,
        softmax=point.softmax,
        uniform_random=point.uniform_random,
        cp_lower=np.quantile(cp, qlo, axis=0),
        cp_upper=np.quantile(cp, qhi, axis=0),
        softmax_lower=np.quantile(sm, qlo, axis=0),
        softmax_upper=np.quantile(sm, qhi, axis=0),
        uniform_lower=np.quantile(un, qlo, axis=0),
        uniform_upper=np.quantile(un, qhi, axis=0),
        cp_minus_uniform_lower=np.quantile(cp - un, qlo, axis=0),
        cp_minus_uniform_upper=np.quantile(cp - un, qhi, axis=0),
        cp_minus_softmax_lower=np.quantile(cp - sm, qlo, axis=0),
        cp_minus_softmax_upper=np.quantile(cp - sm, qhi, axis=0),
    )
