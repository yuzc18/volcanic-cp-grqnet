"""Run the Table-7 calibration-design sensitivity analysis.

Every repeat refits eRF, both Group-C XGBoost regressors, B/C features,
preprocessing, GRQ-Net, and q-hat.  FZI boundaries/labels are inherited from the
single base context and remain frozen.  Full mode implements all manuscript
families; --smoke runs one repeat of each family for CI/runtime validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.pipeline import (  # noqa: E402
    RuntimeOptions,
    build_context,
    context_with_calibration_sample_ids,
    train_final_model,
)
from src.uncertainty.calibration_designs import (  # noqa: E402
    contiguous_blocks_design,
    repeat_seeds,
    stratified_within_well_design,
    whole_well_design,
)
from src.uncertainty.workflow import apply_randomized_aps  # noqa: E402


def _load_cfg() -> dict:
    with open(ROOT / "configs" / "conformal.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _options(smoke: bool, seed: int) -> RuntimeOptions:
    if not smoke:
        return RuntimeOptions(seed=int(seed))
    return RuntimeOptions(
        seed=int(seed),
        erf_n_estimators=1,
        erf_max_depth=4,
        xgb_n_estimators=2,
        max_lith_rows_per_class=4,
        max_epochs=1,
    )


def _designs(base, *, smoke: bool, base_seed: int):
    cfg = _load_cfg()["calibration_designs"]
    dev = base.development
    wells = tuple(base.wells.training_well_names)
    blind = tuple(base.wells.blind_well_names)
    designs = []
    for row in cfg["stratified_fractions"]:
        if smoke and abs(float(row["fraction"]) - 0.20) > 1e-12:
            continue
        reps = 1 if smoke else int(row["repeats"])
        for seed in repeat_seeds(base_seed, reps):
            designs.append(stratified_within_well_design(
                dev, n_cal=int(row["n_cal"]), seed=seed, well_order=wells,
                evaluation_wells=blind,
                name=f"stratified_{int(round(100*float(row['fraction'])))}pct",
                fraction=float(row["fraction"]),
            ))
    row = cfg["contiguous_depth_block"]
    reps = 1 if smoke else int(row["repeats"])
    for seed in repeat_seeds(base_seed, reps):
        designs.append(contiguous_blocks_design(
            dev, n_cal=int(row["n_cal"]), seed=seed, well_order=wells,
            evaluation_wells=blind, name="contiguous_blocks_20pct",
            fraction=float(row["fraction"]),
        ))
    for whole_i, row in enumerate(cfg["whole_well"]):
        if smoke and whole_i > 0:
            continue
        eval_wells = blind if row["evaluation"] == "blind_wells_combined" else (str(row["evaluation"]),)
        designs.append(whole_well_design(
            dev, calibration_well=str(row["calibration_well"]),
            evaluation_wells=eval_wells, seed=base_seed,
            name=f"whole_well_{row['calibration_well']}",
        ))
    return designs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "table7")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if not (args.data_dir / "continuous_logs.csv").exists():
        if args.data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(args.data_dir)
        write_synthetic_inputs(args.data_dir, 20260827)

    base = build_context(args.data_dir, cp_seed=args.seed)
    designs = _designs(base, smoke=args.smoke, base_seed=args.seed)
    rows = []
    index_rows = []
    for i, design in enumerate(designs, 1):
        ctx = context_with_calibration_sample_ids(
            base,
            list(design.calibration_sample_ids),
            extra_upstream_excluded_wells=design.whole_well_calibration,
        )
        final = train_final_model(ctx, options=_options(args.smoke, design.seed))
        eval_frame = final.blind_frame.loc[
            final.blind_frame["well_id"].astype(str).isin(design.evaluation_wells)
        ].reset_index(drop=True)
        app = apply_randomized_aps(
            final.calibration_frame,
            eval_frame,
            alpha=args.alpha,
            rng=np.random.default_rng(design.seed),
        )
        rows.append({
            "design": design.name,
            "kind": design.kind,
            "repeat_seed": design.seed,
            "n_cal": design.n_cal,
            "evaluation_wells": "|".join(design.evaluation_wells),
            "q_hat": app.q_hat,
            "coverage": app.report.marginal_coverage,
            "mean_set_size": app.report.mean_set_size,
            "singleton_rate": app.report.singleton_rate,
            "modeling_n": len(final.prepared.modeling_rows),
            "early_stopping_n": len(final.prepared.early_stop_local_idx),
            "gradient_training_n": len(final.prepared.gradient_local_idx),
        })
        for sid in design.calibration_sample_ids:
            index_rows.append({
                "design": design.name,
                "repeat_seed": design.seed,
                "sample_id": sid,
                "synthetic_demo": "synthetic" in str(args.data_dir).lower(),
            })
        print(f"[{i:02d}/{len(designs):02d}] {design.name} seed={design.seed} n={design.n_cal} coverage={app.report.marginal_coverage:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "table7_repeat_results.csv", index=False)
    pd.DataFrame(index_rows).to_csv(args.output_dir / "calibration_indices.csv", index=False)
    summary = (
        detail.groupby(["design", "kind", "n_cal", "evaluation_wells"], as_index=False)
        .agg(
            repeats=("coverage", "size"),
            coverage_mean=("coverage", "mean"),
            coverage_min=("coverage", "min"),
            coverage_max=("coverage", "max"),
            mean_set_size=("mean_set_size", "mean"),
            singleton_rate=("singleton_rate", "mean"),
        )
    )
    summary.to_csv(args.output_dir / "table7_summary.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps({
        "alpha": args.alpha,
        "base_seed": args.seed,
        "smoke": args.smoke,
        "repeat_seed_policy": "base_seed + repeat_index",
        "repeat_seed_scope": "calibration selection, stochastic downstream/upstream training where applicable, and APS randomization",
        "fzi_boundaries_refit_per_design": False,
        "all_predictive_components_refit_per_design": True,
        "synthetic_demo": "synthetic" in str(args.data_dir).lower(),
    }, indent=2), encoding="utf-8")
    print("Table-7 calibration-design run: PASS")


if __name__ == "__main__":
    main()
