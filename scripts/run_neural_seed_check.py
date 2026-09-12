#!/usr/bin/env python
"""Repeat GRQ-Net, MLP, and TabNet over seeds 20260827-20260829.

This implements the neural-model repeat described in Section 3.6. The same
leakage-safe prepared folds are reused; only neural initialization/training seed
changes. Upstream feature construction remains tied to the primary analysis
pipeline unless the caller regenerates the prepared folds separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.models.baselines import build_neural_baseline  # noqa: E402
from src.pipeline import RuntimeOptions, build_context, prepare_outer_fold  # noqa: E402
from src.training.baseline_neural import evaluate_neural_baseline, train_neural_baseline  # noqa: E402
from src.training.baseline_runner import load_baseline_config  # noqa: E402
from src.training.metrics import compute_metrics, summarize_cv  # noqa: E402
from src.training.trainer import TrainConfig, predict_grqnet, train_one_fold  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--models", nargs="*", default=["grqnet", "mlp", "tabnet"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default="outputs/neural_seed_check.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "continuous_logs.csv").exists():
        if data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(f"Input files not found under {data_dir}; see data/README.md.")
        write_synthetic_inputs(data_dir, 20260827)

    with open(ROOT / "configs" / "seeds.yaml", encoding="utf-8") as f:
        seeds = yaml.safe_load(f)["seed"]["neural_repeats"]
    bcfg = load_baseline_config()
    opt = RuntimeOptions(
        erf_n_estimators=1 if args.smoke else 300,
        xgb_n_estimators=5 if args.smoke else 300,
        max_lith_rows_per_class=6 if args.smoke else None,
        max_epochs=1 if args.smoke else 200,
    )
    ctx = build_context(data_dir, cp_seed=opt.seed)
    prepared = [prepare_outer_fold(ctx, f, options=opt) for f in ctx.partition.folds]
    out = {}
    for seed in seeds:
        out[str(seed)] = {}
        for name in args.models:
            metrics = []
            for p in prepared:
                g, e = p.gradient_local_idx, p.early_stop_local_idx
                if name == "grqnet":
                    fit = train_one_fold(
                        p.X_outer_train[g], p.y_outer_train[g],
                        p.X_outer_train[e], p.y_outer_train[e],
                        train_cfg=TrainConfig(max_epochs=opt.max_epochs), seed=int(seed),
                        class_weight_labels=p.y_outer_train,
                    )
                    pred = predict_grqnet(fit.model, p.X_outer_val)
                    metrics.append(compute_metrics(p.y_outer_val, pred.predictions))
                else:
                    build = build_neural_baseline(name, bcfg)
                    fit = train_neural_baseline(
                        build.model,
                        p.X_outer_train[g], p.y_outer_train[g],
                        p.X_outer_train[e], p.y_outer_train[e],
                        class_weight_labels=p.y_outer_train,
                        train_cfg=TrainConfig(max_epochs=opt.max_epochs), seed=int(seed),
                    )
                    metrics.append(evaluate_neural_baseline(fit, p.X_outer_val, p.y_outer_val))
            out[str(seed)][name] = summarize_cv(metrics)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
