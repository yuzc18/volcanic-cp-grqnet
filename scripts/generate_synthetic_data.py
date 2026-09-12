"""Generate fully synthetic inputs for the revision pipeline.

The proprietary study data are not included in this repository. This script
creates synthetic files that satisfy the documented input schemas so that code
can be exercised without disclosing or approximating the study measurements.
The generated values are *not* intended to reproduce any manuscript result.

Outputs
-------
data/synthetic/continuous_logs.csv
    Continuous 0.125 m LWD grid used to derive Groups A/D and to feed upstream
    models.
data/synthetic/core_samples.csv
    Core-matched rows carrying synthetic porosity/permeability targets.
data/synthetic/lithology_corpus.csv
    Synthetic lithology-labelled corpus used to exercise the Group-B eRF
    fitting pipeline.
data/synthetic/manifest.json
    Generation seed and explicit synthetic-data disclaimer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_wells_config  # noqa: E402

GRID_SPACING_M = 0.125
RAW_CURVES = ["GR", "CNL", "DEN", "AC", "RLA5"]
N_LITH_CLASSES = 18


def _continuous_depth_grid(depth_min: float, depth_max: float) -> np.ndarray:
    """Return a 0.125 m grid contained within the configured well interval."""
    n = int(np.floor((depth_max - depth_min) / GRID_SPACING_M)) + 1
    return depth_min + np.arange(n, dtype=float) * GRID_SPACING_M


def _smooth_noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    raw = rng.normal(0.0, scale, size=n)
    kernel = np.array([0.2, 0.6, 0.2])
    return np.convolve(raw, kernel, mode="same")


def _generate_logs_for_well(
    well_name: str,
    depths: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic continuous logs with weak depth structure."""
    n = len(depths)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(0, 6 * np.pi, n)
    well_shift = (sum(ord(c) for c in well_name) % 17 - 8) / 10.0

    gr = 100 + 18 * np.sin(t + phase) + _smooth_noise(rng, n, 12) + 2 * well_shift
    cnl = 16 + 3.5 * np.cos(0.8 * t + phase / 2) + _smooth_noise(rng, n, 2.2)
    den = 2.52 - 0.015 * (cnl - 16) + _smooth_noise(rng, n, 0.035)
    ac = 72 + 1.5 * (cnl - 16) + _smooth_noise(rng, n, 4.5)
    rla5 = np.exp(3.4 - 0.012 * (gr - 100) + _smooth_noise(rng, n, 0.28))

    return pd.DataFrame(
        {
            "well_id": well_name,
            "depth_m": np.round(depths, 3),
            "GR": np.clip(gr, 20, 250),
            "CNL": np.clip(cnl, 1, 45),
            "DEN": np.clip(den, 2.0, 3.0),
            "AC": np.clip(ac, 40, 140),
            "RLA5": np.clip(rla5, 1, 500),
        }
    )


def _choose_core_rows(
    logs: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select interior grid rows so centered and causal windows are available."""
    candidate = np.arange(2, len(logs) - 2)
    if n_samples > len(candidate):
        raise ValueError("Configured sample count exceeds available interior log-grid rows.")
    chosen = rng.choice(candidate, size=n_samples, replace=False)
    return np.sort(chosen)


def _petrophysics_from_logs(
    core_logs: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deliberately noisy synthetic core porosity/permeability."""
    cnl = core_logs["CNL"].to_numpy()
    den = core_logs["DEN"].to_numpy()
    gr = core_logs["GR"].to_numpy()

    porosity = 5.2 + 0.22 * (cnl - np.median(cnl)) - 4.0 * (den - np.median(den))
    porosity += rng.normal(0, 1.5, size=len(core_logs))
    porosity = np.clip(porosity, 1.0, 18.0)

    log10_k = -1.5 + 0.11 * (porosity - 5.0) - 0.002 * (gr - 100)
    log10_k += rng.normal(0, 0.65, size=len(core_logs))
    permeability = np.clip(10 ** log10_k, 0.001, 30.0)
    return porosity, permeability


def _lithology_labels(logs: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Create synthetic 18-class labels without reproducing study distributions."""
    score = (
        0.025 * logs["GR"].to_numpy()
        + 0.12 * logs["CNL"].to_numpy()
        - 1.5 * logs["DEN"].to_numpy()
        + 0.004 * logs["AC"].to_numpy()
        + rng.normal(0, 1.2, size=len(logs))
    )
    ranks = pd.qcut(score, q=N_LITH_CLASSES, labels=False, duplicates="drop")
    labels = np.asarray(ranks, dtype=int)
    # qcut can return fewer bins for pathological tiny inputs; this generator
    # uses long grids, but keep the output in the declared class range.
    return np.clip(labels, 0, N_LITH_CLASSES - 1)


def generate_synthetic_inputs(seed: int = 20260827) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate continuous logs, core samples, and a lithology corpus."""
    rng = np.random.default_rng(seed)
    wells_cfg = load_wells_config()

    all_logs: list[pd.DataFrame] = []
    all_core: list[pd.DataFrame] = []
    all_lith: list[pd.DataFrame] = []

    for well in wells_cfg.training_wells + wells_cfg.blind_wells:
        depths = _continuous_depth_grid(well.depth_min, well.depth_max)
        logs = _generate_logs_for_well(well.name, depths, rng)
        logs.insert(0, "log_id", [f"SYN-{well.name}-L{i:05d}" for i in range(len(logs))])

        core_idx = _choose_core_rows(logs, well.n_samples, rng)
        core_logs = logs.iloc[core_idx].copy()
        porosity, permeability = _petrophysics_from_logs(core_logs, rng)
        sample_ids = [f"SYN-{well.name}-C{i:04d}" for i in range(well.n_samples)]

        core = core_logs[["well_id", "depth_m"]].copy()
        core.insert(0, "sample_id", sample_ids)
        core["porosity_pct"] = porosity
        core["permeability_mD"] = permeability

        depth_to_sample = dict(zip(core["depth_m"], core["sample_id"]))
        lith = logs[["well_id", "depth_m", *RAW_CURVES]].copy()
        lith.insert(0, "lith_record_id", [f"SYN-{well.name}-R{i:05d}" for i in range(len(lith))])
        lith["source_sample_id"] = lith["depth_m"].map(depth_to_sample).fillna("")
        lith["lithology_class_id"] = _lithology_labels(lith, rng)

        all_logs.append(logs)
        all_core.append(core)
        all_lith.append(lith)

    continuous_logs = pd.concat(all_logs, ignore_index=True)
    core_samples = pd.concat(all_core, ignore_index=True)
    lithology_corpus = pd.concat(all_lith, ignore_index=True)

    assert len(core_samples) == wells_cfg.totals["total_samples"]
    assert np.allclose(
        continuous_logs.groupby("well_id")["depth_m"].diff().dropna().to_numpy(),
        GRID_SPACING_M,
        atol=1e-9,
    )
    assert core_samples[["sample_id", "well_id", "depth_m"]].duplicated().sum() == 0
    return continuous_logs, core_samples, lithology_corpus


def write_synthetic_inputs(output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs, core, lith = generate_synthetic_inputs(seed=seed)

    header = "# SYNTHETIC DEMONSTRATION DATA ONLY - NOT STUDY DATA\n"
    for name, df in [
        ("continuous_logs.csv", logs),
        ("core_samples.csv", core),
        ("lithology_corpus.csv", lith),
    ]:
        with open(output_dir / name, "w", encoding="utf-8", newline="") as f:
            f.write(header)
            df.to_csv(f, index=False)

    manifest = {
        "synthetic": True,
        "seed": seed,
        "grid_spacing_m": GRID_SPACING_M,
        "core_sample_rows": int(len(core)),
        "continuous_log_rows": int(len(logs)),
        "lithology_corpus_rows": int(len(lith)),
        "disclaimer": (
            "Synthetic demonstration data only. Values are not derived from the study data "
            "and are not intended to reproduce numerical results reported in the manuscript."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic",
    )
    args = parser.parse_args()
    write_synthetic_inputs(args.output_dir, args.seed)
    print(f"Synthetic inputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
