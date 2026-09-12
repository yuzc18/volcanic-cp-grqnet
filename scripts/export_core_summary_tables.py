#!/usr/bin/env python3
"""Export manuscript-facing Table 1, Table 4, and Table 9-style CSVs.

Table 1 is generated from the public well/config metadata. Table 4 is generated
from supplied core-format data after the active FZI label construction. Table 9(a/b)
are generated when OOF/blind prediction CSVs containing the standard prediction
columns are supplied. Numerical outputs on synthetic data are demonstrations only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.analysis.tables import classification_summary, coverage_summary, petrophysical_summary  # noqa: E402
from src.pipeline import build_context  # noqa: E402


def _table1() -> pd.DataFrame:
    cfg = yaml.safe_load((ROOT / "configs" / "wells.yaml").read_text(encoding="utf-8"))
    rows = []
    for role_key, role_name in (("training_wells", "Development"), ("blind_wells", "Blind Test")):
        for w in cfg[role_key]:
            rows.append({
                "well": w["name"], "role": role_name, "samples": w["n_samples"],
                "interval_m": f"{w['depth_min']:.1f}-{w['depth_max']:.1f}",
                "dominant_lithology": w.get("dominant_lithology"),
                "permeability_range_mD": f"{w['perm_min_mD']}-{w['perm_max_mD']}",
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "tables")
    ap.add_argument("--oof-predictions", type=Path, default=None)
    ap.add_argument("--blind-predictions", type=Path, default=None)
    args = ap.parse_args()

    if not (args.data_dir / "continuous_logs.csv").exists():
        if args.data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(args.data_dir)
        write_synthetic_inputs(args.data_dir, 20260827)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _table1().to_csv(args.output_dir / "table1_well_overview.csv", index=False)
    ctx = build_context(args.data_dir)
    petrophysical_summary(ctx.labeled_core).to_csv(args.output_dir / "table4_petrophysical_summary.csv", index=False)

    if args.oof_predictions or args.blind_predictions:
        class_rows = []
        cov_frames = []
        for label, path in (("Development wells (OOF)", args.oof_predictions), ("Blind wells combined", args.blind_predictions)):
            if path is None:
                continue
            f = pd.read_csv(path)
            class_rows.append(classification_summary(f, label))
            if "cp_covered" in f.columns:
                cov_frames.append(coverage_summary(f, label))
            if label == "Blind wells combined" and "well_id" in f.columns:
                for well, g in f.groupby("well_id", sort=False):
                    class_rows.append(classification_summary(g, str(well)))
                    if "cp_covered" in g.columns:
                        cov_frames.append(coverage_summary(g, str(well)))
        pd.DataFrame(class_rows).to_csv(args.output_dir / "table9a_classification.csv", index=False)
        if cov_frames:
            pd.concat(cov_frames, ignore_index=True).to_csv(args.output_dir / "table9b_coverage.csv", index=False)
    print("Core summary table export: PASS")


if __name__ == "__main__":
    main()
