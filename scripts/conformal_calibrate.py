"""Run the primary randomized non-empty APS analysis of the finalized manuscript.

The script refits the leakage-safe five outer folds and final model, computes a
separate q-hat from the fixed calibration subset for every outer fold, computes
the deployment q-hat from the final model, and applies the randomized boundary
rule once using NumPy default_rng(seed=20260827).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
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


def _report_dict(report):
    return {
        "marginal_coverage": report.marginal_coverage,
        "mean_set_size": report.mean_set_size,
        "singleton_rate": report.singleton_rate,
        "set_size_distribution": report.set_size_distribution,
        "per_class_coverage": report.per_class_coverage,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "conformal_main")
    p.add_argument("--split-metadata", type=Path, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if not (args.data_dir / "continuous_logs.csv").exists():
        if args.data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(args.data_dir)
        write_synthetic_inputs(args.data_dir, 20260827)

    _, folds, final = run_main_experiment(
        args.data_dir,
        options=_options(args.smoke),
        include_final_model=True,
        split_metadata_path=args.split_metadata,
    )
    assert final is not None
    cp = run_main_conformal(folds, final, alpha=args.alpha, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cp.oof_frame.to_csv(args.output_dir / "oof_conformal_predictions.csv", index=False)
    cp.blind_result.frame.to_csv(args.output_dir / "blind_conformal_predictions.csv", index=False)
    payload = {
        "alpha": args.alpha,
        "random_seed": args.seed,
        "randomization": "randomized_APS_with_nonempty_top1_guard",
        "fold_qhat": list(cp.fold_thresholds),
        "oof": _report_dict(
            # Recompute from concatenated rows only for human-readable summary.
            type(cp.fold_results[0].report)(
                marginal_coverage=float(cp.oof_frame["cp_covered"].mean()),
                mean_set_size=float(cp.oof_frame["cp_set_size"].mean()),
                singleton_rate=float((cp.oof_frame["cp_set_size"] == 1).mean()),
                set_size_distribution={
                    int(k): int(v) for k, v in cp.oof_frame["cp_set_size"].value_counts().sort_index().items()
                },
                per_class_coverage={
                    int(c): float(g["cp_covered"].mean())
                    for c, g in cp.oof_frame.groupby("quality_class_id")
                },
            )
        ),
        "deployment_qhat": cp.blind_result.q_hat,
        "blind": _report_dict(cp.blind_result.report),
        "synthetic_demo": "synthetic" in str(args.data_dir).lower(),
    }
    (args.output_dir / "conformal_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("Randomized non-empty APS primary analysis: PASS")
    print("  fold qhat:", ", ".join(f"{x:.4f}" for x in cp.fold_thresholds))
    print(f"  deployment qhat: {cp.blind_result.q_hat:.4f}")
    print(f"  synthetic OOF coverage: {payload['oof']['marginal_coverage']:.4f}")
    print(f"  synthetic blind coverage: {cp.blind_result.report.marginal_coverage:.4f}")
    if args.smoke:
        print("  NOTE: smoke/synthetic values are not manuscript-result reproductions.")


if __name__ == "__main__":
    main()
