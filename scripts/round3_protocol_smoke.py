"""Fast end-to-end Round-3 protocol smoke test on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.pipeline import RuntimeOptions, run_main_experiment  # noqa: E402


def main() -> None:
    data_dir = ROOT / "data" / "synthetic"
    if not (data_dir / "continuous_logs.csv").exists():
        write_synthetic_inputs(data_dir, 20260827)
    options = RuntimeOptions(
        erf_n_estimators=1,
        erf_max_depth=4,
        xgb_n_estimators=3,
        max_lith_rows_per_class=5,
        max_epochs=1,
    )
    ctx, folds, final = run_main_experiment(data_dir, options=options, include_final_model=True)
    assert len(folds) == 5
    assert sum(len(r.prepared.outer_val_rows) for r in folds) == 712
    assert [len(r.prepared.early_stop_local_idx) for r in folds] == [88, 90, 76, 89, 86]
    assert [len(r.prepared.gradient_local_idx) for r in folds] == [496, 508, 430, 498, 487]
    assert final is not None
    assert len(final.prepared.gradient_local_idx) == 605
    assert len(final.prepared.early_stop_local_idx) == 107
    assert len(final.prepared.blind_rows) == 369
    print("Round-3 protocol smoke test: PASS")
    print("  outer validation sizes: 128 / 114 / 206 / 125 / 139")
    print("  outer early-stop sizes: 88 / 90 / 76 / 89 / 86")
    print("  final split: 605 gradient + 107 early-stop")
    print("  NOTE: synthetic predictions are not manuscript-result reproductions.")


if __name__ == "__main__":
    main()
