#!/usr/bin/env python
"""Run Round-6 statistical, figure, SHAP and latency analyses.

The command operates on the leakage-safe fitted predictions from Rounds 3--5.
With the repository's synthetic data, every result is a runnable demonstration
only and is not expected to reproduce manuscript numerical values.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.analysis.benchmark import benchmark_end_to_end, prepare_latency_assets  # noqa: E402
from src.analysis.conformal_analysis import (  # noqa: E402
    accuracy_by_set_size,
    coverage_curve_from_folds,
    set_size_by_class,
    verification_block_bootstrap,
)
from src.analysis.figures import (  # noqa: E402
    figure6_petrophysics,
    figure8_conformal_behavior,
    figure9_blind_profile,
    figure10_interpretability,
    figure11_fzi_prediction_sets,
    figure12_verification_budget,
)
from src.analysis.interpretability import compute_foldwise_shap  # noqa: E402
from src.analysis.statistics import (  # noqa: E402
    clopper_pearson_interval,
    thinning_analysis,
    within_well_coverage_block_interval,
)
from src.pipeline import RuntimeOptions, run_main_experiment  # noqa: E402
from src.uncertainty.workflow import run_main_conformal  # noqa: E402


def _options(smoke: bool) -> RuntimeOptions:
    if not smoke:
        return RuntimeOptions()
    return RuntimeOptions(
        erf_n_estimators=1,
        erf_max_depth=4,
        xgb_n_estimators=3,
        max_lith_rows_per_class=5,
        max_epochs=1,
    )


def _merge_scientific_columns(pred: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample_id", "well_id", "depth_m", "porosity_pct", "permeability_mD", "FZI_um",
        "GR", "CNL", "DEN", "AC", "RLA5",
    ]
    meta = source[cols].copy()
    out = pred.merge(meta, on=["sample_id", "well_id"], how="left", validate="one_to_one")
    if out["depth_m"].isna().any():
        raise AssertionError("Prediction/source merge lost rows.")
    return out


def _coverage_summary(frame: pd.DataFrame) -> dict:
    rows = {}
    for key, g in [("marginal", frame), *[(f"class_{int(c)}", x) for c, x in frame.groupby("quality_class_id")]]:
        k = int(g["cp_covered"].astype(bool).sum())
        n = int(len(g))
        rows[key] = {"covered": k, "total": n, **clopper_pearson_interval(k, n).to_dict()}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "round6")
    ap.add_argument("--split-metadata", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()

    if not (args.data_dir / "continuous_logs.csv").exists():
        if args.data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(args.data_dir)
        write_synthetic_inputs(args.data_dir, args.seed)

    options = _options(args.smoke)
    context, folds, final = run_main_experiment(
        args.data_dir,
        options=options,
        include_final_model=True,
        split_metadata_path=args.split_metadata,
    )
    assert final is not None
    cp = run_main_conformal(folds, final, alpha=0.05, seed=args.seed)

    out = args.output_dir
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    oof = _merge_scientific_columns(cp.oof_frame, context.development)
    blind = _merge_scientific_columns(cp.blind_result.frame, context.blind)
    oof.to_csv(out / "oof_primary_predictions.csv", index=False)
    blind.to_csv(out / "blind_primary_predictions.csv", index=False)
    context.labeled_core.to_csv(out / "labeled_core_for_analysis.csv", index=False)

    curve = coverage_curve_from_folds(folds, seed=args.seed)
    curve.as_frame().to_csv(out / "fig8_coverage_curve.csv", index=False)
    set_size_by_class(oof).to_csv(out / "fig8_set_size_by_class.csv", index=False)
    accuracy_by_set_size(oof).to_csv(out / "fig8_accuracy_by_set_size.csv", index=False)

    # Statistical checks: Clopper-Pearson, dependence-aware blind intervals,
    # and no-retraining thinning on existing predictions.
    n_boot = 20 if args.smoke else 1000
    stats = {
        "oof_clopper_pearson": _coverage_summary(oof),
        "blind_clopper_pearson": _coverage_summary(blind),
        "blind_within_well_block_bootstrap": {},
        "oof_thinning": [x.to_dict() for x in thinning_analysis(oof)],
        "blind_thinning": [x.to_dict() for x in thinning_analysis(blind)],
        "ace_30_point_grid": float(curve.ace),
    }
    for well, g in blind.groupby("well_id", sort=False):
        stats["blind_within_well_block_bootstrap"][str(well)] = within_well_coverage_block_interval(
            g,
            n_resamples=n_boot,
            block_length_m=2.0,
            seed=args.seed,
        ).to_dict()
    (out / "statistical_checks.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Figures whose inputs are available from the main GRQ-Net/CP workflow.
    figure6_petrophysics(context.labeled_core, context.labeler.boundaries_, output=figures / "Fig06_petrophysics.png")
    figure8_conformal_behavior(curve, oof, output=figures / "Fig08_conformal_behavior.png")
    figure9_blind_profile(blind, q_hat=cp.blind_result.q_hat, well_id="CS9", output=figures / "Fig09_CS9_profile.png")
    figure11_fzi_prediction_sets(oof, context.labeler.boundaries_, output=figures / "Fig11_fzi_prediction_sets.png")

    verify = verification_block_bootstrap(
        oof,
        random_draws=10 if args.smoke else 200,
        n_resamples=10 if args.smoke else 1000,
        block_length_m=2.0,
        seed=args.seed,
    )
    verify_frame = verify.as_frame()
    verify_frame.to_csv(out / "fig12_verification_curve.csv", index=False)
    figure12_verification_budget(verify, output=figures / "Fig12_verification_budget.png")
    idx10 = int(np.argmin(np.abs(verify.budget_fraction - 0.10)))
    stats["verification_budget_10pct"] = {
        "budget_count": int(np.floor(0.10 * len(oof))),
        "cp_guided_recovery": float(verify.cp_guided[idx10]),
        "softmax_recovery": float(verify.softmax[idx10]),
        "uniform_recovery": float(verify.uniform_random[idx10]),
        "cp_minus_uniform": float(verify.cp_guided[idx10] - verify.uniform_random[idx10]),
        "cp_minus_uniform_95pct": [float(verify.cp_minus_uniform_lower[idx10]), float(verify.cp_minus_uniform_upper[idx10])],
        "cp_minus_softmax": float(verify.cp_guided[idx10] - verify.softmax[idx10]),
        "cp_minus_softmax_95pct": [float(verify.cp_minus_softmax_lower[idx10]), float(verify.cp_minus_softmax_upper[idx10])],
    }
    (out / "statistical_checks.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    shap_done = False
    if not args.skip_shap:
        shap_summary = compute_foldwise_shap(
            folds,
            background_size=5 if args.smoke else 100,
            seed=args.seed,
            max_eval_rows_per_fold=2 if args.smoke else None,
        )
        shap_summary.feature_importance.to_csv(out / "fig10_feature_shap.csv", index=False)
        shap_summary.group_importance.to_csv(out / "fig10_group_shap.csv", index=False)
        shap_summary.group_weight_summary.to_csv(out / "fig10_group_weight_summary.csv", index=False)
        figure10_interpretability(
            shap_summary.feature_importance,
            shap_summary.group_importance,
            oof,
            output=figures / "Fig10_interpretability.png",
        )
        shap_done = True

    benchmark_done = False
    if not args.skip_benchmark:
        assets = prepare_latency_assets(
            context,
            final,
            q_hat=cp.blind_result.q_hat,
            options=options,
        )
        latency = benchmark_end_to_end(
            assets,
            warmup_iterations=2 if args.smoke else 100,
            measured_iterations=5 if args.smoke else 1000,
            seed=args.seed,
        )
        payload = latency.to_dict()
        payload["runtime_environment"] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "reference_environment_is_manuscript_specific": True,
        }
        payload["synthetic_demo"] = "synthetic" in str(args.data_dir).lower()
        (out / "latency_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        benchmark_done = True

    manifest = {
        "seed": args.seed,
        "synthetic_demo": "synthetic" in str(args.data_dir).lower(),
        "note": "Synthetic/demo numerical values are not manuscript-result reproductions.",
        "figures_generated": ["6", "8", "9", "11", "12"] + (["10"] if shap_done else []),
        "figure7_requires_baseline_oof_predictions": True,
        "shap_completed": shap_done,
        "benchmark_completed": benchmark_done,
        "bootstrap_replicates": n_boot,
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
