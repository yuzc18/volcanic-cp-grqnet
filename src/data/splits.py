"""Manuscript-faithful data partitions for the finalized manuscript.

The public repository does not contain proprietary sample identifiers.  It does,
however, implement the *procedure* and the published cell counts exactly:

* five development wells (890 core-matched samples) and two blind wells;
* a fixed 178-sample CP calibration set drawn within (well, quality-class)
  cells using the Table 3(b) quotas;
* 712 non-calibration development samples for downstream modelling;
* leave-one-development-well-out outer evaluation in the manuscript order;
* a class-stratified 15% internal early-stopping subset (ceil rounding) drawn
  only from each outer-training subset, and analogously from all 712 rows for
  the final blind-well model.

No calibration row, outer validation row, or blind-well row is ever returned as
an internal gradient-training row.  The actual study sample IDs cannot be
reconstructed without the proprietary data and are therefore not fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import CONFIG_DIR, WellsConfig, load_wells_config

# Quality IDs follow src.data.labels: High=0, Medium=1, Low=2.
CLASS_IDS = (0, 1, 2)
CLASS_NAMES_BY_ID = {0: "High", 1: "Medium", 2: "Low"}


@dataclass(frozen=True)
class CalibrationSplit:
    """Absolute positional indices into the 890-row development frame."""

    cv_pool_idx: np.ndarray
    cp_calib_idx: np.ndarray


@dataclass(frozen=True)
class CVFold:
    """One explicit leave-one-development-well-out outer fold.

    ``train_idx`` and ``val_idx`` are absolute positional indices into the
    890-row development frame, not local indices into the 712-row pool.
    """

    fold_id: int  # 1..5, matching Table 3(b)
    validation_well: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_wells: tuple[str, ...]
    val_wells: tuple[str, ...]


@dataclass(frozen=True)
class EarlyStoppingSplit:
    """Partition of a modelling subset into gradient and checkpoint rows."""

    gradient_idx: np.ndarray  # positions local to the supplied subset
    early_stop_idx: np.ndarray  # positions local to the supplied subset


@dataclass(frozen=True)
class FullPartition:
    """Development/blind roles plus calibration and five outer folds."""

    development_df: pd.DataFrame
    blind_df: pd.DataFrame
    calibration: CalibrationSplit
    folds: tuple[CVFold, ...]


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def _load_data_protocol(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else CONFIG_DIR / "data.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def calibration_cell_quotas(path: str | Path | None = None) -> dict[str, dict[int, int]]:
    """Return the published Table 3(b) (well x class) calibration quotas."""
    cfg = _load_data_protocol(path)
    raw = cfg["counts"]["main_calibration_per_well_class"]
    out: dict[str, dict[int, int]] = {}
    name_to_id = {v: k for k, v in CLASS_NAMES_BY_ID.items()}
    for well, counts in raw.items():
        out[str(well)] = {name_to_id[name]: int(n) for name, n in counts.items()}
    return out


def outer_fold_spec(path: str | Path | None = None) -> list[dict]:
    return list(_load_data_protocol(path)["outer_folds"])


# ---------------------------------------------------------------------------
# Basic roles
# ---------------------------------------------------------------------------
def split_development_blind(
    core_df: pd.DataFrame,
    *,
    wells_cfg: WellsConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the seven study wells into development and blind roles."""
    wells_cfg = wells_cfg or load_wells_config()
    if "well_id" not in core_df.columns:
        raise ValueError("core_df must contain well_id.")
    observed = set(core_df["well_id"].astype(str))
    unknown = observed - set(wells_cfg.all_well_names)
    if unknown:
        raise ValueError(f"Unknown study well IDs: {sorted(unknown)}")

    dev = core_df.loc[core_df["well_id"].isin(wells_cfg.training_well_names)].copy()
    blind = core_df.loc[core_df["well_id"].isin(wells_cfg.blind_well_names)].copy()
    dev = dev.reset_index(drop=True)
    blind = blind.reset_index(drop=True)
    if len(dev) != wells_cfg.totals["training_samples"]:
        raise ValueError(
            f"Development rows={len(dev)}; expected {wells_cfg.totals['training_samples']}."
        )
    if len(blind) != wells_cfg.totals["blind_samples"]:
        raise ValueError(
            f"Blind rows={len(blind)}; expected {wells_cfg.totals['blind_samples']}."
        )
    return dev, blind


# ---------------------------------------------------------------------------
# Fixed CP calibration subset
# ---------------------------------------------------------------------------
def make_calibration_split(
    development_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    seed: int = 20260827,
    quotas: Mapping[str, Mapping[int, int]] | None = None,
    wells_cfg: WellsConfig | None = None,
) -> CalibrationSplit:
    """Sample the fixed-design calibration set using exact Table 3(b) quotas.

    The proprietary paper-specific row identities are not available.  Given a
    user-supplied dataset, this function deterministically selects rows using the
    published per-well/per-class cell sizes and ``seed``.  On the original
    private data, the authors should supply the preserved indices instead if
    exact historical row identity is required.
    """
    wells_cfg = wells_cfg or load_wells_config()
    quotas = dict(quotas or calibration_cell_quotas())
    labels = np.asarray(labels, dtype=int)
    n = len(development_df)
    if n != wells_cfg.totals["training_samples"]:
        raise ValueError(f"Expected 890 development rows, got {n}.")
    if len(labels) != n:
        raise ValueError("labels length does not match development_df.")
    if "well_id" not in development_df.columns:
        raise ValueError("development_df must contain well_id.")
    if set(development_df["well_id"]) & set(wells_cfg.blind_well_names):
        raise AssertionError("Blind rows entered calibration selection.")

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for well in wells_cfg.training_well_names:
        if well not in quotas:
            raise ValueError(f"No calibration quota configured for {well}.")
        well_mask = development_df["well_id"].to_numpy() == well
        for cls in CLASS_IDS:
            q = int(quotas[well].get(cls, 0))
            candidates = np.flatnonzero(well_mask & (labels == cls))
            if len(candidates) < q:
                raise ValueError(
                    f"Calibration cell ({well}, {CLASS_NAMES_BY_ID[cls]}) needs {q} rows "
                    f"but only {len(candidates)} are available."
                )
            if q:
                picked = rng.choice(candidates, size=q, replace=False)
                chosen.extend(int(i) for i in picked)

    calib = np.array(sorted(chosen), dtype=np.int64)
    if len(np.unique(calib)) != len(calib):
        raise AssertionError("Calibration selection contains duplicate rows.")
    pool = np.setdiff1d(np.arange(n, dtype=np.int64), calib, assume_unique=True)

    expected_n = int(_load_data_protocol()["counts"]["main_calibration"])
    if len(calib) != expected_n:
        raise AssertionError(f"Calibration set has {len(calib)} rows; expected {expected_n}.")
    if len(pool) != n - expected_n:
        raise AssertionError("CV pool size mismatch.")
    if np.intersect1d(calib, pool).size:
        raise AssertionError("Calibration and CV pool overlap.")

    # Published marginal class counts: 51 / 49 / 78.
    cfg_counts = _load_data_protocol()["counts"]["main_calibration_class_counts"]
    expected_by_class = {
        0: int(cfg_counts["High"]),
        1: int(cfg_counts["Medium"]),
        2: int(cfg_counts["Low"]),
    }
    observed_by_class = {cls: int(np.sum(labels[calib] == cls)) for cls in CLASS_IDS}
    if observed_by_class != expected_by_class:
        raise AssertionError(
            f"Calibration class counts {observed_by_class} do not match {expected_by_class}."
        )
    return CalibrationSplit(cv_pool_idx=pool, cp_calib_idx=calib)


def make_calibration_split_from_sample_ids(
    development_df: pd.DataFrame,
    labels: np.ndarray,
    calibration_sample_ids: Iterable[str],
) -> CalibrationSplit:
    """Use an authoritative fixed calibration-ID list when one is available.

    This path exists for a genuinely authoritative anonymized sample-ID file
    supplied by the authors/user. It reproduces that 178-row assignment exactly
    and never derives or guesses IDs from aggregate counts.
    """
    if "sample_id" not in development_df.columns:
        raise ValueError("development_df must contain sample_id.")
    wanted = {str(x) for x in calibration_sample_ids}
    sample_ids = development_df["sample_id"].astype(str).to_numpy()
    missing = wanted - set(sample_ids)
    if missing:
        raise ValueError(f"Authoritative calibration IDs not found in input: {sorted(missing)[:5]}")
    calib = np.flatnonzero(np.isin(sample_ids, list(wanted))).astype(np.int64)
    if len(calib) != 178:
        raise ValueError(f"Authoritative calibration list selects {len(calib)} rows; expected 178.")
    pool = np.setdiff1d(np.arange(len(development_df), dtype=np.int64), calib, assume_unique=True)

    # Validate the published Table 3(b) design rather than silently accepting
    # an incompatible file.
    quotas = calibration_cell_quotas()
    y = np.asarray(labels, dtype=int)
    for well, qmap in quotas.items():
        for cls, expected in qmap.items():
            observed = int(np.sum((development_df.iloc[calib]["well_id"].to_numpy() == well) & (y[calib] == cls)))
            if observed != expected:
                raise ValueError(
                    f"Authoritative split cell ({well}, {CLASS_NAMES_BY_ID[cls]})={observed}; expected {expected}."
                )
    return CalibrationSplit(cv_pool_idx=pool, cp_calib_idx=np.sort(calib))


# ---------------------------------------------------------------------------
# Explicit outer leave-one-well-out folds
# ---------------------------------------------------------------------------
def make_well_level_folds(
    development_df: pd.DataFrame,
    cv_pool_idx: np.ndarray,
    *,
    wells_cfg: WellsConfig | None = None,
    spec: list[dict] | None = None,
) -> list[CVFold]:
    """Build the five outer folds in the exact Table 3(b) well order."""
    wells_cfg = wells_cfg or load_wells_config()
    spec = spec or outer_fold_spec()
    pool = np.asarray(cv_pool_idx, dtype=np.int64)
    all_pool_set = set(pool.tolist())
    folds: list[CVFold] = []
    for row in spec:
        fold_id = int(row["fold"])
        val_well = str(row["validation_well"])
        is_val = development_df.iloc[pool]["well_id"].to_numpy() == val_well
        val_idx = pool[is_val]
        train_idx = pool[~is_val]
        train_wells = tuple(
            w for w in wells_cfg.training_well_names if w != val_well
        )
        if len(val_idx) != int(row["validation_n"]):
            raise AssertionError(
                f"Fold {fold_id} {val_well}: validation n={len(val_idx)}, "
                f"expected {row['validation_n']}."
            )
        if len(train_idx) != int(row["outer_training_n"]):
            raise AssertionError(
                f"Fold {fold_id}: outer training n={len(train_idx)}, "
                f"expected {row['outer_training_n']}."
            )
        if set(val_idx.tolist()) & set(train_idx.tolist()):
            raise AssertionError("Outer train/validation row overlap.")
        if set(val_idx.tolist()) | set(train_idx.tolist()) != all_pool_set:
            raise AssertionError("Outer fold does not partition the 712-row pool.")
        observed_train_wells = set(development_df.iloc[train_idx]["well_id"])
        if val_well in observed_train_wells:
            raise AssertionError("Held-out validation well leaked into outer training rows.")
        folds.append(
            CVFold(
                fold_id=fold_id,
                validation_well=val_well,
                train_idx=train_idx,
                val_idx=val_idx,
                train_wells=train_wells,
                val_wells=(val_well,),
            )
        )
    return folds


# ---------------------------------------------------------------------------
# Internal 15% early-stopping split
# ---------------------------------------------------------------------------
def make_early_stopping_split(
    labels: np.ndarray,
    *,
    fraction: float = 0.15,
    seed: int = 20260827,
    expected_early_n: int | None = None,
) -> EarlyStoppingSplit:
    """Class-stratified random internal hold-out with ceil rounding."""
    y = np.asarray(labels, dtype=int)
    n = len(y)
    if n < 2:
        raise ValueError("Need at least two rows for an internal split.")
    n_early = int(ceil(n * fraction))
    if expected_early_n is not None and n_early != int(expected_early_n):
        raise AssertionError(
            f"ceil({fraction} * {n})={n_early}, expected {expected_early_n}."
        )
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=n_early, random_state=int(seed)
    )
    gradient, early = next(splitter.split(np.zeros(n), y))
    gradient = np.sort(gradient.astype(np.int64))
    early = np.sort(early.astype(np.int64))
    if np.intersect1d(gradient, early).size:
        raise AssertionError("Gradient and early-stopping rows overlap.")
    if len(gradient) + len(early) != n:
        raise AssertionError("Internal split does not cover all outer-training rows.")
    return EarlyStoppingSplit(gradient_idx=gradient, early_stop_idx=early)


def make_outer_early_stopping_split(
    development_df: pd.DataFrame,
    labels: np.ndarray,
    fold: CVFold,
    *,
    seed: int = 20260827,
) -> EarlyStoppingSplit:
    """Internal split for one outer fold, checked against Table 3(b)."""
    row = next(r for r in outer_fold_spec() if int(r["fold"]) == fold.fold_id)
    y = np.asarray(labels, dtype=int)[fold.train_idx]
    split = make_early_stopping_split(
        y,
        seed=seed,
        expected_early_n=int(row["early_stopping_n"]),
    )
    if len(split.gradient_idx) != int(row["gradient_training_n"]):
        raise AssertionError("Gradient-training count does not match Table 3(b).")
    return split


def make_final_early_stopping_split(
    labels_712: np.ndarray,
    *,
    seed: int = 20260827,
) -> EarlyStoppingSplit:
    """605 gradient + 107 early-stop rows for the final 712-row model."""
    cfg = _load_data_protocol()["final_neural_fit"]
    split = make_early_stopping_split(
        np.asarray(labels_712, dtype=int),
        fraction=float(cfg["early_stopping_fraction"]),
        seed=seed,
        expected_early_n=int(cfg["early_stopping_n"]),
    )
    if len(split.gradient_idx) != int(cfg["gradient_training_n"]):
        raise AssertionError("Final gradient-training count mismatch.")
    return split


# ---------------------------------------------------------------------------
# One-call role partition
# ---------------------------------------------------------------------------
def build_full_partition(
    core_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    cp_seed: int = 20260827,
    wells_cfg: WellsConfig | None = None,
    calibration_sample_ids: Iterable[str] | None = None,
) -> FullPartition:
    """Build development/blind roles, fixed calibration design, and outer folds."""
    wells_cfg = wells_cfg or load_wells_config()
    development, blind = split_development_blind(core_df, wells_cfg=wells_cfg)

    # ``core_df`` and development are sorted by the standard loader; select the
    # development labels by well rather than assuming the blind rows are last.
    dev_mask = core_df["well_id"].isin(wells_cfg.training_well_names).to_numpy()
    dev_labels = np.asarray(labels, dtype=int)[dev_mask]
    if len(dev_labels) != len(development):
        raise AssertionError("Development label alignment failed.")

    if calibration_sample_ids is None:
        calibration = make_calibration_split(
            development,
            dev_labels,
            seed=cp_seed,
            wells_cfg=wells_cfg,
        )
    else:
        calibration = make_calibration_split_from_sample_ids(
            development, dev_labels, calibration_sample_ids
        )
    folds = make_well_level_folds(
        development,
        calibration.cv_pool_idx,
        wells_cfg=wells_cfg,
    )
    return FullPartition(
        development_df=development,
        blind_df=blind,
        calibration=calibration,
        folds=tuple(folds),
    )
