"""CPU inference benchmarking for the six-stage manuscript pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import pandas as pd
import torch

from src.data.io import LWD_COLUMNS
from src.features.group_a import RawLogClipper
from src.features.group_d import GROUP_D_COLUMNS, build_group_d_continuous, match_group_d_to_core
from src.features.uft import UFTPreprocessor
from src.pipeline import ExperimentContext, FinalModelRun, RuntimeOptions
from src.training.upstream import crossfit_training_proxies, fit_upstream_models
from src.uncertainty.conformal import build_prediction_sets_detailed

STAGE_NAMES = (
    "input_preparation_three_sigma_clipping_for_A_D",
    "erf_18_class_lithology",
    "xgboost_regressors_x2",
    "group_d_uft_and_frozen_robust_scaling",
    "grqnet",
    "conformal_set_construction",
)


@dataclass
class LatencyAssets:
    raw_window: pd.DataFrame
    raw_values: np.ndarray
    current_row: pd.DataFrame
    clipper: RawLogClipper
    preprocessor: UFTPreprocessor
    erf: object
    group_c: object
    grqnet: object
    q_hat: float


@dataclass(frozen=True)
class LatencySummary:
    stage_median_ms: dict[str, float]
    stage_p95_ms: dict[str, float]
    total_median_ms: float
    total_p95_ms: float
    throughput_samples_per_second: float
    warmup_iterations: int
    measured_iterations: int

    def to_dict(self) -> dict:
        return {
            "stages": {
                name: {
                    "median_ms": float(self.stage_median_ms[name]),
                    "p95_ms": float(self.stage_p95_ms[name]),
                }
                for name in STAGE_NAMES
            },
            "total": {
                "median_ms": float(self.total_median_ms),
                "p95_ms": float(self.total_p95_ms),
                "throughput_samples_per_second": float(self.throughput_samples_per_second),
            },
            "warmup_iterations": int(self.warmup_iterations),
            "measured_iterations": int(self.measured_iterations),
            "total_is_complete_iteration_wall_clock": True,
        }


def _causal_d_from_clipped_window(clipped: np.ndarray) -> np.ndarray:
    """Build the 25 causal D values from three clipped rows ending at current."""
    if clipped.shape != (3, 5):
        raise ValueError("Expected a 3x5 clipped LWD window ending at the current sample.")
    vals: list[float] = []
    for j in range(5):
        x2, x1, x0 = clipped[:, j]
        vals.extend(
            [
                float(x1),
                float(x2),  # causal replacement occupying the next1 slot
                float(np.mean([x2, x1, x0])),
                float(np.std([x2, x1, x0], ddof=1)),
                float(x0 - x1),
            ]
        )
    out = np.asarray(vals, dtype=float).reshape(1, 25)
    if len(GROUP_D_COLUMNS) != out.shape[1]:
        raise AssertionError("Group-D width changed unexpectedly.")
    return out


def prepare_latency_assets(
    context: ExperimentContext,
    final_run: FinalModelRun,
    *,
    q_hat: float,
    options: RuntimeOptions | None = None,
) -> LatencyAssets:
    """Fit frozen inference assets outside the timed loop.

    The benchmark uses the strictly causal Group-D definition.  Its scaler is
    fitted on the 712 modeling rows using leakage-safe inner-cross-fitted B/C
    training proxies.  Deployment eRF/XGBoost models are fitted on those same
    712 non-calibration development rows and exclude both blind wells.
    """
    options = options or RuntimeOptions()
    modeling = context.modeling_rows.copy()
    train_proxy = crossfit_training_proxies(
        modeling,
        context.inputs.lithology_corpus,
        calibration_sample_ids=context.calibration_sample_ids,
        always_exclude_wells=context.wells.blind_well_names,
        well_order=context.wells.training_well_names,
        erf_n_estimators=options.erf_n_estimators,
        erf_max_depth=options.erf_max_depth,
        erf_random_state=options.erf_random_state,
        xgb_random_state=options.seed,
        xgb_overrides=options.xgb_overrides,
        max_lith_rows_per_class=options.max_lith_rows_per_class,
    )
    clipper = RawLogClipper().fit(modeling)
    d_cont = build_group_d_continuous(context.inputs.continuous_logs, clipper=clipper, causal=True)
    d_model = match_group_d_to_core(modeling, d_cont)
    pre = UFTPreprocessor().fit_on(
        modeling,
        train_proxy.group_b,
        train_proxy.group_c,
        d_model,
        raw_clipper=clipper,
    )
    upstream = fit_upstream_models(
        modeling,
        context.inputs.lithology_corpus,
        calibration_sample_ids=context.calibration_sample_ids,
        exclude_lithology_wells=context.wells.blind_well_names,
        purpose="latency_deployment_upstream",
        target_wells=context.wells.blind_well_names,
        erf_n_estimators=options.erf_n_estimators,
        erf_max_depth=options.erf_max_depth,
        erf_random_state=options.erf_random_state,
        xgb_random_state=options.seed,
        xgb_overrides=options.xgb_overrides,
        max_lith_rows_per_class=options.max_lith_rows_per_class,
    )

    # Select the first blind core sample whose matched log row has two preceding
    # grid samples. Synthetic generation guarantees this; the explicit check
    # keeps private-data runs honest.
    blind = final_run.prepared.blind_rows.copy()
    logs = context.inputs.continuous_logs
    raw_window = None
    for _, row in blind.sort_values(["well_id", "depth_m"]).iterrows():
        g = logs.loc[logs["well_id"].astype(str) == str(row["well_id"])].sort_values("depth_m").reset_index(drop=True)
        matches = np.flatnonzero(g["log_id"].astype(str).to_numpy() == str(row["log_id"]))
        if len(matches) == 1 and matches[0] >= 2:
            raw_window = g.iloc[matches[0] - 2 : matches[0] + 1][["well_id", "depth_m", *LWD_COLUMNS]].copy()
            break
    if raw_window is None:
        raise ValueError("No blind sample has the two preceding grid rows required by causal Group D.")

    return LatencyAssets(
        raw_window=raw_window,
        raw_values=raw_window[LWD_COLUMNS].to_numpy(dtype=float),
        current_row=raw_window.iloc[[-1]].copy(),
        clipper=clipper,
        preprocessor=pre,
        erf=upstream.erf,
        group_c=upstream.group_c,
        grqnet=final_run.train_result.model,
        q_hat=float(q_hat),
    )


def _one_iteration(assets: LatencyAssets, rng: np.random.Generator, *, collect: bool) -> tuple[np.ndarray, float]:
    stage = np.zeros(6, dtype=np.int64)
    t_total0 = perf_counter_ns()

    t0 = perf_counter_ns()
    clipped = assets.clipper.transform_array(assets.raw_values)
    t1 = perf_counter_ns()
    stage[0] = t1 - t0

    t0 = perf_counter_ns()
    b = assets.erf.predict_proba(assets.current_row)
    t1 = perf_counter_ns()
    stage[1] = t1 - t0

    t0 = perf_counter_ns()
    c = assets.group_c.predict(assets.current_row).as_array()
    t1 = perf_counter_ns()
    stage[2] = t1 - t0

    t0 = perf_counter_ns()
    d = _causal_d_from_clipped_window(clipped)
    assert assets.preprocessor.scaler_a is not None
    assert assets.preprocessor.scaler_c is not None
    assert assets.preprocessor.scaler_d is not None
    a = assets.preprocessor.scaler_a.transform(clipped[[-1]])
    cs = assets.preprocessor.scaler_c.transform(c)
    ds = assets.preprocessor.scaler_d.transform(d)
    x = np.concatenate([a, b, cs, ds], axis=1).astype(np.float32)
    t1 = perf_counter_ns()
    stage[3] = t1 - t0

    t0 = perf_counter_ns()
    assets.grqnet.eval()
    with torch.no_grad():
        logits = assets.grqnet(torch.from_numpy(x))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    t1 = perf_counter_ns()
    stage[4] = t1 - t0

    t0 = perf_counter_ns()
    u = rng.uniform(0.0, 1.0, size=1)
    build_prediction_sets_detailed(
        probs,
        assets.q_hat,
        uniforms=u,
        randomized=True,
        always_retain_top=True,
    )
    t1 = perf_counter_ns()
    stage[5] = t1 - t0

    total = float(perf_counter_ns() - t_total0)
    return stage, total


def benchmark_end_to_end(
    assets: LatencyAssets,
    *,
    warmup_iterations: int = 100,
    measured_iterations: int = 1000,
    seed: int = 20260827,
) -> LatencySummary:
    """Benchmark the complete batch-one pipeline on one CPU thread."""
    if warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("Invalid benchmark iteration counts.")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits setting interop threads before parallel work has
        # begun. The command-line benchmark sets it at process start.
        pass
    rng = np.random.default_rng(seed)
    for _ in range(int(warmup_iterations)):
        _one_iteration(assets, rng, collect=False)

    stages = np.empty((int(measured_iterations), 6), dtype=float)
    totals = np.empty(int(measured_iterations), dtype=float)
    for i in range(int(measured_iterations)):
        s, total = _one_iteration(assets, rng, collect=True)
        stages[i] = s / 1e6
        totals[i] = total / 1e6
    med = np.median(stages, axis=0)
    p95 = np.percentile(stages, 95, axis=0)
    total_med = float(np.median(totals))
    total_p95 = float(np.percentile(totals, 95))
    return LatencySummary(
        stage_median_ms={name: float(med[i]) for i, name in enumerate(STAGE_NAMES)},
        stage_p95_ms={name: float(p95[i]) for i, name in enumerate(STAGE_NAMES)},
        total_median_ms=total_med,
        total_p95_ms=total_p95,
        throughput_samples_per_second=float(1000.0 / total_med) if total_med > 0 else float("inf"),
        warmup_iterations=int(warmup_iterations),
        measured_iterations=int(measured_iterations),
    )


def benchmark_torch_classifier(
    model,
    X: np.ndarray,
    *,
    warmup_iterations: int = 100,
    measured_iterations: int = 1000,
) -> dict[str, float]:
    """Standalone batch-one classifier latency helper for Table 6(a)."""
    torch.set_num_threads(1)
    x = torch.from_numpy(np.asarray(X[:1], dtype=np.float32))
    model.eval()
    with torch.no_grad():
        for _ in range(int(warmup_iterations)):
            model(x)
        times = np.empty(int(measured_iterations), dtype=float)
        for i in range(int(measured_iterations)):
            t0 = perf_counter_ns()
            model(x)
            times[i] = (perf_counter_ns() - t0) / 1e6
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
