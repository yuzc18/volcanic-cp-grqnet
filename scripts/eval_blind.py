"""Fit the active final model, calibrate deployment APS, and evaluate blind wells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.pipeline import RuntimeOptions, build_context, train_final_model  # noqa: E402
from src.uncertainty.workflow import (  # noqa: E402
    DEPLOYMENT_UNIT,
    apply_randomized_aps,
    conformal_substream,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "blind_cp")
    p.add_argument("--split-metadata", type=Path, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if not (args.data_dir / "continuous_logs.csv").exists():
        if args.data_dir.resolve() != (ROOT / "data" / "synthetic").resolve():
            raise FileNotFoundError(args.data_dir)
        write_synthetic_inputs(args.data_dir, 20260827)

    options = RuntimeOptions()
    if args.smoke:
        options = RuntimeOptions(
            erf_n_estimators=1, erf_max_depth=4, xgb_n_estimators=3,
            max_lith_rows_per_class=5, max_epochs=1,
        )
    ctx = build_context(
        args.data_dir, cp_seed=args.seed, split_metadata_path=args.split_metadata
    )
    final = train_final_model(ctx, options=options)
    # Same named substream as run_main_conformal's deployment step, so this
    # standalone entry point reproduces the main analysis exactly for identical
    # probabilities, sample identities and seed.
    app = apply_randomized_aps(
        final.calibration_frame,
        final.blind_frame,
        alpha=args.alpha,
        rng=conformal_substream(args.seed, DEPLOYMENT_UNIT),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    app.frame.to_csv(args.output_dir / "blind_conformal_predictions.csv", index=False)
    payload = {
        "q_hat": app.q_hat,
        "coverage": app.report.marginal_coverage,
        "mean_set_size": app.report.mean_set_size,
        "singleton_rate": app.report.singleton_rate,
        "synthetic_demo": "synthetic" in str(args.data_dir).lower(),
    }
    (args.output_dir / "blind_cp_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
