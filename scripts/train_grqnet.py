"""Run the manuscript-faithful GRQ-Net split/cross-fitting pipeline.

Research mode uses the canonical paper-facing hyperparameters (300-tree eRF,
300-round Group-C XGBoost, up to 200 neural epochs).  ``--smoke`` is an
explicit synthetic/CI acceleration mode: it keeps the same data roles and
leakage barriers while reducing estimator sizes and epochs.  Smoke outputs are
never manuscript-result reproductions.

No proprietary data are bundled. Supply the documented three-file input
directory or use the synthetic generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.config import load_config, setup_reproducibility  # noqa: E402
from src.pipeline import RuntimeOptions, run_main_experiment  # noqa: E402
from src.training.metrics import compute_metrics, summarize_cv  # noqa: E402


def _options(smoke: bool) -> RuntimeOptions:
    if not smoke:
        return RuntimeOptions()
    return RuntimeOptions(
        erf_n_estimators=1,
        erf_max_depth=4,
        xgb_n_estimators=5,
        max_lith_rows_per_class=6,
        max_epochs=1,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "synthetic",
        help="Directory containing continuous_logs.csv, core_samples.csv, lithology_corpus.csv.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "grqnet",
        help="Local output directory (gitignored).",
    )
    p.add_argument(
        "--split-metadata",
        type=Path,
        default=None,
        help=("Optional authoritative study-specific split file. Its sample_id values must match "
              "core_samples.csv sample_id values; if omitted, the published Table 3(b) cell quotas "
              "are sampled deterministically for demonstration/reconstruction use."),
    )
    p.add_argument("--smoke", action="store_true", help="Fast synthetic/CI execution controls.")
    p.add_argument("--skip-final", action="store_true", help="Run outer CV only.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not (args.data_dir / "continuous_logs.csv").exists():
        if args.data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(
                f"Input files not found under {args.data_dir}; see data/README.md."
            )
        write_synthetic_inputs(args.data_dir, 20260827)

    cfg = load_config()
    setup_reproducibility(cfg)
    options = _options(args.smoke)

    print("=" * 78)
    print("GRQ-Net Round-3 pipeline: outer LOOW + inner upstream cross-fitting")
    print("=" * 78)
    print(f"data_dir: {args.data_dir}")
    if args.smoke:
        print("SMOKE MODE: reduced estimator sizes/epochs; protocol checks only.")
        print("Synthetic outputs are not expected to reproduce manuscript numbers.")

    context, folds, final = run_main_experiment(
        args.data_dir,
        options=options,
        include_final_model=not args.skip_final,
        verbose=args.verbose,
        split_metadata_path=args.split_metadata,
    )

    fold_metrics = [r.metrics for r in folds]
    cv_summary = summarize_cv(fold_metrics)
    oof = pd.concat([r.oof_frame for r in folds], ignore_index=True)
    pooled = compute_metrics(
        oof["quality_class_id"].to_numpy(dtype=int),
        oof["predicted_class_id"].to_numpy(dtype=int),
    )
    cal_by_fold = pd.concat([r.calibration_frame for r in folds], ignore_index=True)

    print("\nOuter-fold row counts:")
    for r in folds:
        p0 = r.prepared
        print(
            f"  fold {p0.fold.fold_id} {p0.fold.validation_well}: "
            f"outer_train={len(p0.outer_train_rows)}, "
            f"gradient={len(p0.gradient_local_idx)}, early={len(p0.early_stop_local_idx)}, "
            f"outer_val={len(p0.outer_val_rows)}, calibration={len(p0.calibration_rows)}"
        )
    print(f"OOF rows: {len(oof)} (expected 712)")
    print(f"Synthetic pooled macro-F1: {pooled.macro_f1:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    cal_by_fold.to_csv(args.output_dir / "fold_calibration_probabilities.csv", index=False)

    payload = {
        "synthetic_demo": "synthetic" in str(args.data_dir).lower(),
        "smoke_mode": bool(args.smoke),
        "fzi_boundaries_um": context.labeler.boundaries_.as_array().tolist(),
        "cv_summary": cv_summary,
        "pooled_oof_metrics": pooled.to_dict(),
        "folds": [
            {
                "fold": r.prepared.fold.fold_id,
                "validation_well": r.prepared.fold.validation_well,
                "outer_training_n": len(r.prepared.outer_train_rows),
                "gradient_training_n": len(r.prepared.gradient_local_idx),
                "early_stopping_n": len(r.prepared.early_stop_local_idx),
                "validation_n": len(r.prepared.outer_val_rows),
                "best_epoch": r.train_result.best_epoch,
                "outer_metrics": r.metrics.to_dict(),
            }
            for r in folds
        ],
    }

    if final is not None:
        final.blind_frame.to_csv(args.output_dir / "blind_predictions.csv", index=False)
        final.calibration_frame.to_csv(
            args.output_dir / "final_calibration_probabilities.csv", index=False
        )
        payload["final_model"] = {
            "modeling_n": len(final.prepared.modeling_rows),
            "gradient_training_n": len(final.prepared.gradient_local_idx),
            "early_stopping_n": len(final.prepared.early_stop_local_idx),
            "calibration_n": len(final.prepared.calibration_rows),
            "blind_n": len(final.prepared.blind_rows),
            "best_epoch": final.train_result.best_epoch,
            "blind_metrics": final.blind_metrics.to_dict(),
        }
        print(
            f"Final model rows: 712 -> {len(final.prepared.gradient_local_idx)} gradient + "
            f"{len(final.prepared.early_stop_local_idx)} early-stop; blind={len(final.prepared.blind_rows)}"
        )

    (args.output_dir / "round3_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"Local outputs written under {args.output_dir} (gitignored).")


if __name__ == "__main__":
    main()
