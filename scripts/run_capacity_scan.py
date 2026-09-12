#!/usr/bin/env python
"""Run the seven post-hoc one-factor-at-a-time GRQ-Net capacity configurations."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402

from src.analysis.capacity import capacity_scan_configs
from src.pipeline import RuntimeOptions, build_context, prepare_outer_fold
from src.training.metrics import summarize_cv, compute_metrics
from src.training.trainer import TrainConfig, train_one_fold, predict_grqnet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default="outputs/capacity_scan.json")
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
    ctx = build_context(args.data_dir, cp_seed=opt.seed)
    prepared = [prepare_outer_fold(ctx, f, options=opt) for f in ctx.partition.folds]
    result = {}
    for label, model_cfg in capacity_scan_configs():
        metrics = []
        for p in prepared:
            g,e=p.gradient_local_idx,p.early_stop_local_idx
            fit=train_one_fold(
                p.X_outer_train[g],p.y_outer_train[g],p.X_outer_train[e],p.y_outer_train[e],
                train_cfg=TrainConfig(max_epochs=opt.max_epochs), model_cfg=model_cfg,
                seed=opt.seed, class_weight_labels=p.y_outer_train,
            )
            pred=predict_grqnet(fit.model,p.X_outer_val)
            metrics.append(compute_metrics(p.y_outer_val,pred.predictions))
        result[label]={"config":asdict(model_cfg),"cv":summarize_cv(metrics)}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
