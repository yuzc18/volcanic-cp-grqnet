#!/usr/bin/env python
"""Run Table-6(a) baseline models on the leakage-safe Round-3 UFT folds.

Default execution uses the manuscript HPO protocol and the full machine-readable
search spaces in ``configs/search_spaces.yaml``. ``--reported-fixed-demo`` is
retained as a fast fixed-configuration demonstration path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402

import pandas as pd
import yaml

from src.pipeline import RuntimeOptions, build_context, prepare_outer_fold, prepare_final_model
from src.training.baseline_non_neural import UnrecoveredSearchSpaceError, load_search_protocol
from src.training.baseline_runner import load_baseline_config, run_neural_on_prepared, run_non_neural_on_prepared, run_final_non_neural_prepared
from src.training.metrics import summarize_cv
from src.training.trainer import TrainConfig

NEURAL = ["mlp", "lstm", "transformer", "tabnet"]
NON_NEURAL = ["lightgbm", "multinomial_logistic_regression", "random_forest", "xgboost", "catboost"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--models", nargs="*", default=NEURAL + NON_NEURAL)
    ap.add_argument("--reported-fixed-demo", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default="outputs/baselines")
    ap.add_argument("--include-final-non-neural", action="store_true", help="Also tune/final-fit non-neural models on 712 rows and evaluate the blind rows.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "continuous_logs.csv").exists():
        if data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(f"Input files not found under {data_dir}; see data/README.md.")
        write_synthetic_inputs(data_dir, 20260827)
    args.data_dir = str(data_dir)

    opt = RuntimeOptions(
        erf_n_estimators=1 if args.smoke else 300,
        xgb_n_estimators=5 if args.smoke else 300,
        max_lith_rows_per_class=6 if args.smoke else None,
        max_epochs=1 if args.smoke else 200,
    )
    baseline_cfg = load_baseline_config()
    search_protocol = load_search_protocol("configs/search_spaces.yaml")
    unavailable = [m for m in args.models if m in NON_NEURAL and not search_protocol["search_protocol"][m].get("available", False)]
    if unavailable and not args.reported_fixed_demo:
        raise UnrecoveredSearchSpaceError(
            "Search spaces are unavailable for: " + ", ".join(unavailable)
        )
    context = build_context(args.data_dir, cp_seed=opt.seed)
    prepared = [prepare_outer_fold(context, f, options=opt) for f in context.partition.folds]
    train_cfg = TrainConfig(max_epochs=opt.max_epochs)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    config_records = []
    prediction_records = []
    for name in args.models:
        fold_results = []
        for p in prepared:
            if name in NEURAL:
                r = run_neural_on_prepared(name, p, baseline_cfg=baseline_cfg, train_cfg=train_cfg, seed=opt.seed)
            else:
                mode = "reported_fixed_demo" if args.reported_fixed_demo else "manuscript_hpo"
                r = run_non_neural_on_prepared(
                    name, p, baseline_cfg=baseline_cfg, search_protocol=search_protocol,
                    seed=opt.seed, mode=mode, smoke=args.smoke,
                )
            fold_results.append(r)
            config_records.append({
                "model": name, "fold": r.fold_id, "validation_well": r.validation_well,
                "selection_engine": r.selection_engine, "selection_metric": "macro_f1",
                "internal_macro_f1": r.internal_macro_f1,
                "selected_params": json.dumps(r.selected_params, sort_keys=True), "seed": opt.seed,
            })
            rows = p.outer_val_rows.reset_index(drop=True)
            for i, yhat in enumerate(r.predictions):
                prediction_records.append({
                    "model": name,
                    "fold": int(r.fold_id),
                    "validation_well": r.validation_well,
                    "sample_id": str(rows.loc[i, "sample_id"]),
                    "well_id": str(rows.loc[i, "well_id"]),
                    "depth_m": float(rows.loc[i, "depth_m"]),
                    "quality_class_id": int(rows.loc[i, "quality_class_id"]),
                    "predicted_class_id": int(yhat),
                })
        summary[name] = summarize_cv([r.metrics for r in fold_results])
    pd.DataFrame(config_records).to_csv(out / "per_fold_configurations.csv", index=False)
    pd.DataFrame(prediction_records).to_csv(out / "oof_predictions.csv", index=False)
    final_summary = {}
    if args.include_final_non_neural:
        pfinal = prepare_final_model(context, options=opt)
        for name in [m for m in args.models if m in NON_NEURAL]:
            mode = "reported_fixed_demo" if args.reported_fixed_demo else "manuscript_hpo"
            r = run_final_non_neural_prepared(name, pfinal, baseline_cfg=baseline_cfg, search_protocol=search_protocol, seed=opt.seed, mode=mode, smoke=args.smoke)
            final_summary[name] = r.metrics.to_dict()
            config_records.append({"model": name, "fold": "final", "validation_well": "CS9+CS10", "selection_engine": r.selection_engine, "selection_metric": "macro_f1", "internal_macro_f1": r.internal_macro_f1, "selected_params": json.dumps(r.selected_params, sort_keys=True), "seed": opt.seed})
        pd.DataFrame(config_records).to_csv(out / "per_fold_configurations.csv", index=False)
        (out / "final_non_neural_blind_metrics.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"outer_cv": summary, "final_non_neural": final_summary}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except UnrecoveredSearchSpaceError as exc:
        raise SystemExit(str(exc))
