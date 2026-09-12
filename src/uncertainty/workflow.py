"""Conformal orchestration for the primary analysis and Table-7 refits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.pipeline import FinalModelRun, OuterFoldRun
from src.uncertainty.conformal import (
    CoverageReport,
    aps_nonconformity_scores,
    build_prediction_sets_detailed,
    compute_threshold,
)


PROB_COLS = ["prob_High", "prob_Medium", "prob_Low"]


@dataclass(frozen=True)
class APSApplication:
    q_hat: float
    report: CoverageReport
    frame: pd.DataFrame
    calibration_scores: np.ndarray


def _report_from_sets(labels: np.ndarray, sets: list[list[int]], sizes: np.ndarray) -> CoverageReport:
    y = np.asarray(labels, dtype=int)
    covered = np.array([int(y[i]) in sets[i] for i in range(len(y))], dtype=bool)
    return CoverageReport(
        marginal_coverage=float(covered.mean()),
        mean_set_size=float(sizes.mean()),
        singleton_rate=float(np.mean(sizes == 1)),
        set_size_distribution={int(s): int(np.sum(sizes == s)) for s in np.unique(sizes)},
        per_class_coverage={
            c: float(covered[y == c].mean()) if np.any(y == c) else float("nan")
            for c in range(3)
        },
    )


def apply_randomized_aps(
    calibration_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    *,
    alpha: float,
    rng: np.random.Generator,
) -> APSApplication:
    """Calibrate and apply randomized non-empty APS using one RNG stream."""
    p_cal = calibration_frame[PROB_COLS].to_numpy(dtype=float)
    y_cal = calibration_frame["quality_class_id"].to_numpy(dtype=int)
    p_eval = evaluation_frame[PROB_COLS].to_numpy(dtype=float)
    y_eval = evaluation_frame["quality_class_id"].to_numpy(dtype=int)

    u_cal = rng.uniform(0.0, 1.0, size=len(calibration_frame))
    scores = aps_nonconformity_scores(p_cal, y_cal, uniforms=u_cal, randomized=True)
    q_hat = compute_threshold(scores, alpha)
    u_eval = rng.uniform(0.0, 1.0, size=len(evaluation_frame))
    details = build_prediction_sets_detailed(
        p_eval,
        q_hat,
        uniforms=u_eval,
        randomized=True,
        always_retain_top=True,
    )
    report = _report_from_sets(y_eval, details.sets, details.sizes)
    covered = np.array([int(y_eval[i]) in details.sets[i] for i in range(len(y_eval))], dtype=bool)

    out = evaluation_frame.copy()
    out["cp_set"] = ["|".join(map(str, s)) for s in details.sets]
    out["cp_set_size"] = details.sizes
    out["cp_covered"] = covered
    out["cp_boundary_removed"] = details.boundary_removed
    out["cp_uniform"] = details.uniforms
    out["q_hat"] = q_hat
    out["alpha"] = float(alpha)
    return APSApplication(q_hat=q_hat, report=report, frame=out, calibration_scores=scores)


@dataclass(frozen=True)
class MainConformalRun:
    fold_results: tuple[APSApplication, ...]
    oof_frame: pd.DataFrame
    blind_result: APSApplication

    @property
    def fold_thresholds(self) -> tuple[float, ...]:
        return tuple(float(x.q_hat) for x in self.fold_results)


def run_main_conformal(
    folds: list[OuterFoldRun],
    final: FinalModelRun,
    *,
    alpha: float = 0.05,
    seed: int = 20260827,
) -> MainConformalRun:
    """Apply fold-specific q-hat values and the final deployment q-hat.

    One NumPy ``default_rng(seed)`` stream supplies every randomized calibration
    score and prediction-boundary step, in manuscript fold order followed by the
    final blind-well evaluation.
    """
    rng = np.random.default_rng(seed)
    fold_apps: list[APSApplication] = []
    for run in folds:
        app = apply_randomized_aps(
            run.calibration_frame, run.oof_frame, alpha=alpha, rng=rng
        )
        app.frame["fold"] = int(run.prepared.fold.fold_id)
        fold_apps.append(app)
    oof = pd.concat([x.frame for x in fold_apps], ignore_index=True)
    blind = apply_randomized_aps(
        final.calibration_frame, final.blind_frame, alpha=alpha, rng=rng
    )
    return MainConformalRun(
        fold_results=tuple(fold_apps), oof_frame=oof, blind_result=blind
    )
