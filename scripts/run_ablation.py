#!/usr/bin/env python
"""Run Table-5 feature ablations on the leakage-safe outer folds.

The manuscript requires LightGBM hyperparameters to be selected separately for
each feature configuration within each outer fold. That manuscript HPO path is
the default and uses ``configs/search_spaces.yaml``. ``--reported-fixed-demo``
remains available as a fast fixed-configuration execution check.

For GRQ-Net ablations, omitted semantic groups and their encoders are removed,
and the network is re-instantiated with the active groups only. All other model
and training settings are unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.analysis.ablation import TABLE5_CONFIGS, grq_ablation_inputs_and_config, select_groups  # noqa: E402
from src.pipeline import RuntimeOptions, build_context, prepare_outer_fold  # noqa: E402
from src.training.baseline_non_neural import (  # noqa: E402
    UnrecoveredSearchSpaceError,
    build_non_neural_estimator,
    fixed_reported_params,
    load_search_protocol,
    tune_non_neural,
)
from src.training.baseline_runner import load_baseline_config  # noqa: E402
from src.training.metrics import compute_metrics, summarize_cv  # noqa: E402
from src.training.trainer import TrainConfig, predict_grqnet, train_one_fold  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--reported-fixed-demo", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default="outputs/ablation.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "continuous_logs.csv").exists():
        if data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(f"Input files not found under {data_dir}; see data/README.md.")
        write_synthetic_inputs(data_dir, 20260827)

    bcfg = load_baseline_config()
    protocol = load_search_protocol(ROOT / "configs" / "search_spaces.yaml")
    opt = RuntimeOptions(
        erf_n_estimators=1 if args.smoke else 300,
        xgb_n_estimators=5 if args.smoke else 300,
        max_lith_rows_per_class=6 if args.smoke else None,
        max_epochs=1 if args.smoke else 200,
    )
    ctx = build_context(data_dir, cp_seed=opt.seed)
    prepared = [prepare_outer_fold(ctx, f, options=opt) for f in ctx.partition.folds]
    result = {}
    for label, groups in TABLE5_CONFIGS.items():
        gm, lm = [], []
        lgb_selections = []
        for p in prepared:
            Xtr_g, grq_cfg = grq_ablation_inputs_and_config(p.X_outer_train, groups)
            Xv_g, _ = grq_ablation_inputs_and_config(p.X_outer_val, groups)
            g, e = p.gradient_local_idx, p.early_stop_local_idx
            fit = train_one_fold(
                Xtr_g[g], p.y_outer_train[g], Xtr_g[e], p.y_outer_train[e],
                train_cfg=TrainConfig(max_epochs=opt.max_epochs), model_cfg=grq_cfg,
                seed=opt.seed, class_weight_labels=p.y_outer_train,
            )
            pred = predict_grqnet(fit.model, Xv_g)
            gm.append(compute_metrics(p.y_outer_val, pred.predictions))

            Xtr_l = select_groups(p.X_outer_train, groups)
            Xv_l = select_groups(p.X_outer_val, groups)
            if args.reported_fixed_demo:
                params = fixed_reported_params("lightgbm", bcfg, smoke=args.smoke)
                model = build_non_neural_estimator("lightgbm", params, seed=opt.seed)
                model.fit(Xtr_l, p.y_outer_train)
                engine, internal = "reported_table3a_fixed_demo_not_hpo", None
            else:
                sel = tune_non_neural(
                    "lightgbm", Xtr_l[g], p.y_outer_train[g], Xtr_l[e], p.y_outer_train[e],
                    protocol=protocol, baseline_cfg=bcfg, seed=opt.seed,
                    refit_X=Xtr_l, refit_y=p.y_outer_train, smoke=args.smoke,
                )
                model, params, engine, internal = sel.model, sel.best_params, sel.engine, sel.internal_macro_f1
            lm.append(compute_metrics(p.y_outer_val, np.asarray(model.predict(Xv_l)).reshape(-1)))
            lgb_selections.append({"fold": p.fold.fold_id, "params": params, "engine": engine, "internal_macro_f1": internal})
        result[label] = {
            "active_dim": int(select_groups(prepared[0].X_outer_train, groups).shape[1]),
            "grqnet": summarize_cv(gm),
            "lightgbm": summarize_cv(lm),
            "lightgbm_selections": lgb_selections,
            "demo_fixed_config": bool(args.reported_fixed_demo),
            "grq_missing_group_strategy": "subset_reinstantiate",
        }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except UnrecoveredSearchSpaceError as exc:
        raise SystemExit(str(exc))
