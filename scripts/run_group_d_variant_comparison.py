"""Centered vs strictly causal Group D under the same folds, seed and protocol.

Section 3.3 defines a strictly causal Group D variant that replaces the three
forward-looking statistics with trailing-window equivalents, and Section 4.3
reports the cross-validation macro-F1 of that variant alongside the centered
default.  This driver produces that comparison.

Both arms rebuild Group D, refit the preprocessing on the rebuilt block and
retrain GRQ-Net on the same five outer folds with the same seed, so the only
difference between them is the Group D window definition.  The latency benchmark
also builds causal features, but it reuses an already-trained model and
therefore cannot stand in for this accuracy comparison.

Example
-------
    python scripts/run_group_d_variant_comparison.py \\
        --output-dir outputs/group_d_variant

Synthetic-data results are demonstration values and are not expected to match
the manuscript.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.pipeline import (  # noqa: E402
    RuntimeOptions,
    build_context,
    prepare_outer_fold,
)
from src.training.metrics import compute_metrics  # noqa: E402
from src.training.trainer import TrainConfig, predict_grqnet, train_one_fold  # noqa: E402

VARIANTS = ("centered", "causal")


def _options(variant: str, *, seed: int, smoke: bool) -> RuntimeOptions:
    if smoke:
        return RuntimeOptions(
            seed=seed, erf_n_estimators=1, erf_max_depth=4, xgb_n_estimators=3,
            max_lith_rows_per_class=5, max_epochs=1, group_d_variant=variant,
        )
    return RuntimeOptions(seed=seed, group_d_variant=variant)


def run_variant(context, variant: str, *, seed: int, smoke: bool) -> list[dict]:
    options = _options(variant, seed=seed, smoke=smoke)
    rows = []
    for fold in context.partition.folds:
        p = prepare_outer_fold(context, fold, options=options)
        g, e = p.gradient_local_idx, p.early_stop_local_idx
        trained = train_one_fold(
            p.X_outer_train[g], p.y_outer_train[g],
            p.X_outer_train[e], p.y_outer_train[e],
            train_cfg=TrainConfig(max_epochs=options.max_epochs),
            seed=seed,
            class_weight_labels=p.y_outer_train,
        )
        pred = predict_grqnet(trained.model, p.X_outer_val)
        m = compute_metrics(p.y_outer_val, pred.predictions)
        rows.append({
            "group_d_variant": variant,
            "fold": int(fold.fold_id),
            "validation_well": fold.validation_well,
            "macro_f1": float(m.macro_f1),
            "accuracy": float(m.accuracy),
            "mcc": float(m.mcc),
        })
        print(f"  {variant:<8} fold {fold.fold_id} ({fold.validation_well}): "
              f"macro-F1 = {m.macro_f1:.4f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "group_d_variant")
    ap.add_argument("--split-metadata", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not (args.data_dir / "continuous_logs.csv").exists():
        if args.data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(args.data_dir)
        write_synthetic_inputs(args.data_dir, args.seed)

    context = build_context(
        args.data_dir, cp_seed=args.seed, split_metadata_path=args.split_metadata
    )

    rows: list[dict] = []
    for variant in VARIANTS:
        rows.extend(run_variant(context, variant, seed=args.seed, smoke=args.smoke))

    frame = pd.DataFrame(rows)
    means = frame.groupby("group_d_variant")["macro_f1"].agg(["mean", "std", "count"])
    centered = float(means.loc["centered", "mean"])
    causal = float(means.loc["causal", "mean"])

    paired = (
        frame.pivot(index="fold", columns="group_d_variant", values="macro_f1")
        .assign(causal_minus_centered_pp=lambda d: 100.0 * (d["causal"] - d["centered"]))
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "group_d_variant_per_fold.csv", index=False)
    paired.to_csv(args.output_dir / "group_d_variant_paired.csv", index=False)
    summary = {
        "seed": args.seed,
        "smoke": args.smoke,
        "synthetic_demo": "synthetic" in str(args.data_dir).lower(),
        "centered_macro_f1_mean": centered,
        "causal_macro_f1_mean": causal,
        "causal_minus_centered_pp": 100.0 * (causal - centered),
        "n_folds": int(means.loc["centered", "count"]),
        "same_folds_and_seed": True,
        "note": (
            "Both arms rebuild Group D, refit preprocessing and retrain GRQ-Net; "
            "synthetic values are demonstration values, not manuscript results."
        ),
    }
    (args.output_dir / "group_d_variant_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
