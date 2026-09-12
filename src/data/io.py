"""Three-file data interface for the revision implementation.

No proprietary measurements are distributed with the repository.  The public
pipeline consumes three user-supplied tables:

``continuous_logs.csv``
    Continuous 0.125 m LWD grid used for Groups A/D and as inputs to the
    upstream models.
``core_samples.csv``
    Core-matched porosity/permeability records used for FZI labels and Group C.
``lithology_corpus.csv``
    Lithology-labelled five-log corpus used to refit the 18-class Group-B eRF.

All loaders accept the synthetic files produced by
``scripts/generate_synthetic_data.py``.  Comment lines beginning with ``#`` are
ignored so the synthetic-data disclaimer can remain physically embedded in the
CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LWD_COLUMNS: list[str] = ["GR", "CNL", "DEN", "AC", "RLA5"]
CONTINUOUS_META: list[str] = ["log_id", "well_id", "depth_m"]
CORE_META: list[str] = ["sample_id", "well_id", "depth_m"]
CORE_TARGETS: list[str] = ["porosity_pct", "permeability_mD"]
LITH_META: list[str] = ["lith_record_id", "well_id", "depth_m", "source_sample_id"]
LITH_TARGET: str = "lithology_class_id"
GRID_SPACING_M: float = 0.125


@dataclass(frozen=True)
class StudyInputs:
    """Validated inputs used by the reconstructed pipeline."""

    continuous_logs: pd.DataFrame
    core_samples: pd.DataFrame
    lithology_corpus: pd.DataFrame


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, comment="#")


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_unique(df: pd.DataFrame, cols: list[str], name: str) -> None:
    dup = df.duplicated(cols, keep=False)
    if bool(dup.any()):
        example = df.loc[dup, cols].head(5).to_dict("records")
        raise ValueError(f"{name} contains duplicate keys {cols}: {example}")


def _require_finite(df: pd.DataFrame, cols: list[str], name: str) -> None:
    arr = df[cols].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains missing or non-finite values in {cols}.")


def load_continuous_logs(
    path: str | Path,
    *,
    expected_spacing_m: float = GRID_SPACING_M,
    spacing_atol: float = 1e-6,
) -> pd.DataFrame:
    """Load and validate the continuous LWD grid.

    The manuscript dataset contains no retained sample with missing/invalid logs.
    The public implementation therefore fails loudly on non-finite inputs rather
    than silently imputing or deleting rows.
    """
    df = _read_csv(path)
    _require_columns(df, CONTINUOUS_META + LWD_COLUMNS, "continuous_logs")
    _require_unique(df, ["log_id"], "continuous_logs")
    _require_unique(df, ["well_id", "depth_m"], "continuous_logs")
    _require_finite(df, ["depth_m", *LWD_COLUMNS], "continuous_logs")

    out = df.copy()
    out["well_id"] = out["well_id"].astype(str)
    out = out.sort_values(["well_id", "depth_m"], kind="stable").reset_index(drop=True)

    for well, g in out.groupby("well_id", sort=False):
        depths = g["depth_m"].to_numpy(dtype=float)
        if len(depths) > 1:
            delta = np.diff(depths)
            if not np.allclose(delta, expected_spacing_m, atol=spacing_atol, rtol=0):
                bad = delta[~np.isclose(delta, expected_spacing_m, atol=spacing_atol, rtol=0)]
                raise ValueError(
                    f"continuous_logs well {well!r} is not a continuous "
                    f"{expected_spacing_m:.3f} m grid; example spacings={bad[:5]!r}."
                )
    return out


def load_core_samples(path: str | Path) -> pd.DataFrame:
    """Load core-matched porosity/permeability records without altering values."""
    df = _read_csv(path)
    _require_columns(df, CORE_META + CORE_TARGETS, "core_samples")
    _require_unique(df, ["sample_id"], "core_samples")
    _require_unique(df, ["well_id", "depth_m"], "core_samples")
    _require_finite(df, ["depth_m", *CORE_TARGETS], "core_samples")

    out = df.copy()
    out["well_id"] = out["well_id"].astype(str)
    if (out["porosity_pct"] <= 0).any() or (out["porosity_pct"] >= 100).any():
        raise ValueError("porosity_pct must be in (0, 100).")
    if (out["permeability_mD"] <= 0).any():
        raise ValueError("permeability_mD must be strictly positive for log10/FZI.")
    return out.sort_values(["well_id", "depth_m"], kind="stable").reset_index(drop=True)


def load_lithology_corpus(path: str | Path, *, n_classes: int = 18) -> pd.DataFrame:
    """Load the labelled corpus used to refit the upstream lithology classifier."""
    df = _read_csv(path)
    _require_columns(df, LITH_META + LWD_COLUMNS + [LITH_TARGET], "lithology_corpus")
    _require_unique(df, ["lith_record_id"], "lithology_corpus")
    _require_finite(df, ["depth_m", *LWD_COLUMNS, LITH_TARGET], "lithology_corpus")

    out = df.copy()
    out["well_id"] = out["well_id"].astype(str)
    out["source_sample_id"] = out["source_sample_id"].fillna("").astype(str)
    labels = out[LITH_TARGET].to_numpy(dtype=int)
    if not np.equal(labels, out[LITH_TARGET].to_numpy(dtype=float)).all():
        raise ValueError("lithology_class_id must contain integer values.")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= n_classes:
        raise ValueError(f"lithology_class_id must be in [0, {n_classes - 1}].")
    out[LITH_TARGET] = labels
    return out.sort_values(["well_id", "depth_m"], kind="stable").reset_index(drop=True)


def load_inputs(
    directory: str | Path,
    *,
    expected_spacing_m: float = GRID_SPACING_M,
) -> StudyInputs:
    """Load the standard three-file input directory."""
    directory = Path(directory)
    return StudyInputs(
        continuous_logs=load_continuous_logs(
            directory / "continuous_logs.csv", expected_spacing_m=expected_spacing_m
        ),
        core_samples=load_core_samples(directory / "core_samples.csv"),
        lithology_corpus=load_lithology_corpus(directory / "lithology_corpus.csv"),
    )


def match_core_to_continuous(
    core_samples: pd.DataFrame,
    continuous_logs: pd.DataFrame,
    *,
    tolerance_m: float = 0.5,
) -> pd.DataFrame:
    """Match each core record to the nearest continuous-grid row within a well.

    The source data were depth-registered before analysis.  This helper exposes
    that registration step without changing the core-measured porosity or
    permeability.  Ties are resolved toward the shallower log-grid row by the
    stable sort order, which keeps the operation deterministic.

    Returns a copy of ``core_samples`` with ``log_id``, ``log_depth_m``,
    ``depth_offset_m`` and the five raw LWD curves appended.
    """
    _require_columns(core_samples, CORE_META + CORE_TARGETS, "core_samples")
    _require_columns(continuous_logs, CONTINUOUS_META + LWD_COLUMNS, "continuous_logs")

    chunks: list[pd.DataFrame] = []
    for well, core_w in core_samples.groupby("well_id", sort=False):
        logs_w = continuous_logs.loc[continuous_logs["well_id"] == well].copy()
        if logs_w.empty:
            raise ValueError(f"No continuous logs found for core well {well!r}.")
        logs_w = logs_w.sort_values("depth_m", kind="stable")
        depths = logs_w["depth_m"].to_numpy(dtype=float)
        rows = []
        for _, row in core_w.sort_values("depth_m", kind="stable").iterrows():
            d = float(row["depth_m"])
            pos = int(np.searchsorted(depths, d, side="left"))
            cand = []
            if pos < len(depths):
                cand.append(pos)
            if pos > 0:
                cand.append(pos - 1)
            # Deterministic: distance first, then shallower depth.
            best = min(cand, key=lambda j: (abs(depths[j] - d), depths[j]))
            offset = float(depths[best] - d)
            if abs(offset) > tolerance_m + 1e-12:
                raise ValueError(
                    f"Core sample {row['sample_id']!r} at {d:.3f} m has no LWD "
                    f"match within ±{tolerance_m:.3f} m."
                )
            log_row = logs_w.iloc[best]
            merged = row.to_dict()
            merged["log_id"] = log_row["log_id"]
            merged["log_depth_m"] = float(log_row["depth_m"])
            merged["depth_offset_m"] = offset
            for c in LWD_COLUMNS:
                merged[c] = float(log_row[c])
            rows.append(merged)
        chunks.append(pd.DataFrame(rows))

    out = pd.concat(chunks, ignore_index=True)
    if out["sample_id"].duplicated().any():
        raise AssertionError("Core-to-log matching produced duplicate sample IDs.")
    return out


# ---------------------------------------------------------------------------
# Backward-compatible alias used by pre-Round-2 scripts.
# ---------------------------------------------------------------------------
def load_dataset(path: str | Path, *_, **__) -> pd.DataFrame:
    """Compatibility shim for legacy one-table scripts.

    The reconstructed repository no longer defines a measurement-bearing
    one-table public format.  This function accepts a ``core_samples.csv`` path
    and returns the validated core table so stale imports fail less abruptly;
    full modeling code should use :func:`load_inputs` instead.
    """
    return load_core_samples(path)
