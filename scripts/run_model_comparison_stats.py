#!/usr/bin/env python
"""Compute manuscript model-comparison intervals from existing OOF predictions.

Inputs are prediction tables only; models are never refit inside the statistical
procedure. The well is the primary unit for the paired t interval (n=5), while
a paired well-stratified 2.0 m moving-block bootstrap is reported as a
sample-level sensitivity check.
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

from src.analysis.statistics import paired_model_block_bootstrap_difference, paired_t_interval  # noqa: E402
from src.training.metrics import compute_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-predictions", type=Path, required=True)
    ap.add_argument("--grq-oof", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/model_comparison_intervals.json"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    base = pd.read_csv(args.baseline_predictions)
    grq = pd.read_csv(args.grq_oof)[["sample_id", "well_id", "depth_m", "quality_class_id", "predicted_class_id"]].rename(
        columns={"predicted_class_id": "pred_grqnet"}
    )
    result = {}
    for model, g in base.groupby("model", sort=True):
        b = g[["sample_id", "predicted_class_id"]].rename(columns={"predicted_class_id": "pred_baseline"})
        m = grq.merge(b, on="sample_id", how="inner", validate="one_to_one")
        if len(m) != len(grq):
            raise ValueError(f"{model}: baseline predictions do not cover the full GRQ-Net OOF set.")
        diffs = []
        by_well = []
        for well, w in m.groupby("well_id", sort=False):
            mg = compute_metrics(w["quality_class_id"].to_numpy(int), w["pred_grqnet"].to_numpy(int)).macro_f1
            mb = compute_metrics(w["quality_class_id"].to_numpy(int), w["pred_baseline"].to_numpy(int)).macro_f1
            d = float(mg - mb)
            diffs.append(d)
            by_well.append({"well_id": str(well), "grqnet_macro_f1": mg, "baseline_macro_f1": mb, "difference": d})
        if len(diffs) != 5:
            raise ValueError(f"{model}: expected five held-out wells, got {len(diffs)}.")
        t_int = paired_t_interval(diffs)
        block = paired_model_block_bootstrap_difference(
            m,
            true_col="quality_class_id",
            pred_a_col="pred_grqnet",
            pred_b_col="pred_baseline",
            metric="macro_f1",
            n_resamples=args.bootstrap,
            block_length_m=2.0,
            seed=args.seed,
        )
        result[str(model)] = {
            "per_well": by_well,
            "paired_t_interval": t_int.to_dict(),
            "paired_2m_block_bootstrap_interval": block.to_dict(),
            "positive_wells": int(np.sum(np.asarray(diffs) > 0)),
            "n_wells": 5,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
