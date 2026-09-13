"""Calibration-design generators for Table 7.

The manuscript specifies the design families and total calibration sizes. This
module applies those published selection procedures to supplied data and records
the resulting sample IDs for each run.

Those recorded IDs belong to the data supplied to the run -- synthetic inputs, or
the user's own -- and are written to that run's output directory. The repository
itself ships no row-level split manifest: the published metadata are the
aggregate fold sizes and the well-by-class calibration quotas of Table 3(b)
(``metadata/fold_assignments.csv``), plus a schema for an optional
study-specific row mapping (``metadata/split_schema.csv``). The historical
sample-level assignment of the 178 calibration rows is not included.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationDesign:
    name: str
    kind: str
    seed: int
    calibration_sample_ids: tuple[str, ...]
    evaluation_wells: tuple[str, ...]
    whole_well_calibration: tuple[str, ...] = ()
    fraction: float | None = None

    @property
    def n_cal(self) -> int:
        return len(self.calibration_sample_ids)


def repeat_seeds(base_seed: int, repeats: int) -> tuple[int, ...]:
    """Deterministic fixed repeat seeds used by the released implementation."""
    return tuple(int(base_seed) + i for i in range(int(repeats)))


def _largest_remainder(total: int, weights: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    capacities = np.asarray(capacities, dtype=int)
    if total < 0 or total > int(capacities.sum()):
        raise ValueError("allocation total is incompatible with capacities.")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("allocation weights must have positive total mass.")
    ideal = total * weights / weights.sum()
    out = np.minimum(np.floor(ideal).astype(int), capacities)
    remaining = total - int(out.sum())
    remainders = ideal - np.floor(ideal)
    while remaining > 0:
        eligible = np.flatnonzero(out < capacities)
        if len(eligible) == 0:
            raise RuntimeError("no capacity remains for allocation.")
        # Stable tie-break by original order.
        j = int(eligible[np.argmax(remainders[eligible])])
        out[j] += 1
        remainders[j] = -1.0
        remaining -= 1
        if np.all(remainders[eligible] < 0) and remaining > 0:
            remainders = ideal - np.floor(ideal)
            remainders[out >= capacities] = -1.0
    return out


def _well_targets(development: pd.DataFrame, wells: tuple[str, ...], n_cal: int) -> dict[str, int]:
    counts = np.array([int(np.sum(development["well_id"].astype(str).to_numpy() == w)) for w in wells])
    alloc = _largest_remainder(n_cal, counts.astype(float), counts)
    return {w: int(n) for w, n in zip(wells, alloc)}


def stratified_within_well_design(
    development: pd.DataFrame,
    *,
    n_cal: int,
    seed: int,
    well_order: tuple[str, ...],
    evaluation_wells: tuple[str, ...],
    name: str,
    fraction: float | None = None,
) -> CalibrationDesign:
    """Class-stratified sampling within each development well.

    Integer cell counts are obtained by largest-remainder allocation so the
    published total ``n_cal`` is respected exactly.
    """
    rng = np.random.default_rng(seed)
    target_by_well = _well_targets(development, well_order, n_cal)
    chosen: list[str] = []
    for well in well_order:
        g = development.loc[development["well_id"].astype(str) == well]
        target = target_by_well[well]
        class_ids = np.array([0, 1, 2], dtype=int)
        class_counts = np.array([int(np.sum(g["quality_class_id"].to_numpy(dtype=int) == c)) for c in class_ids])
        class_alloc = _largest_remainder(target, class_counts.astype(float), class_counts)
        for cls, take in zip(class_ids, class_alloc):
            ids = g.loc[g["quality_class_id"].to_numpy(dtype=int) == cls, "sample_id"].astype(str).to_numpy()
            if take:
                picked = rng.choice(ids, size=int(take), replace=False)
                chosen.extend(map(str, picked))
    if len(chosen) != n_cal or len(set(chosen)) != n_cal:
        raise AssertionError("stratified calibration design size/uniqueness mismatch.")
    return CalibrationDesign(
        name=name,
        kind="stratified",
        seed=int(seed),
        calibration_sample_ids=tuple(sorted(chosen)),
        evaluation_wells=evaluation_wells,
        fraction=fraction,
    )


def contiguous_blocks_design(
    development: pd.DataFrame,
    *,
    n_cal: int,
    seed: int,
    well_order: tuple[str, ...],
    evaluation_wells: tuple[str, ...],
    name: str,
    fraction: float | None = None,
) -> CalibrationDesign:
    """Select one contiguous depth-ordered core-sample block per development well."""
    rng = np.random.default_rng(seed)
    target_by_well = _well_targets(development, well_order, n_cal)
    chosen: list[str] = []
    for well in well_order:
        g = development.loc[development["well_id"].astype(str) == well].sort_values("depth_m", kind="stable")
        target = target_by_well[well]
        if target > len(g):
            raise ValueError(f"block target {target} exceeds rows in {well}.")
        start = int(rng.integers(0, len(g) - target + 1))
        chosen.extend(g.iloc[start : start + target]["sample_id"].astype(str).tolist())
    if len(chosen) != n_cal or len(set(chosen)) != n_cal:
        raise AssertionError("contiguous calibration design size/uniqueness mismatch.")
    return CalibrationDesign(
        name=name,
        kind="contiguous_depth_blocks",
        seed=int(seed),
        calibration_sample_ids=tuple(chosen),
        evaluation_wells=evaluation_wells,
        fraction=fraction,
    )


def whole_well_design(
    development: pd.DataFrame,
    *,
    calibration_well: str,
    evaluation_wells: tuple[str, ...],
    seed: int,
    name: str,
) -> CalibrationDesign:
    ids = development.loc[
        development["well_id"].astype(str) == str(calibration_well), "sample_id"
    ].astype(str).tolist()
    if not ids:
        raise ValueError(f"calibration well {calibration_well!r} has no development rows.")
    return CalibrationDesign(
        name=name,
        kind="whole_well",
        seed=int(seed),
        calibration_sample_ids=tuple(ids),
        evaluation_wells=tuple(map(str, evaluation_wells)),
        whole_well_calibration=(str(calibration_well),),
    )
