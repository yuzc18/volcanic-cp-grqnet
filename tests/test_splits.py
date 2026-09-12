"""Round-3 tests for the published split protocol."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.generate_synthetic_data import write_synthetic_inputs
from src.config import load_wells_config
from src.data.io import load_inputs, match_core_to_continuous
from src.data.labels import fit_development_labels
from src.data.splits import (
    build_full_partition,
    calibration_cell_quotas,
    make_calibration_split_from_sample_ids,
    make_final_early_stopping_split,
    make_outer_early_stopping_split,
)


@pytest.fixture(scope="module")
def split_context(tmp_path_factory):
    d = tmp_path_factory.mktemp("round3_split_inputs")
    write_synthetic_inputs(d, 20260827)
    inputs = load_inputs(d)
    wells = load_wells_config()
    core = match_core_to_continuous(inputs.core_samples, inputs.continuous_logs)
    _, labeled = fit_development_labels(core, wells.training_well_names)
    part = build_full_partition(
        labeled,
        labeled["quality_class_id"].to_numpy(dtype=int),
        cp_seed=20260827,
        wells_cfg=wells,
    )
    return wells, labeled, part


def test_fixed_calibration_exact_table3b_cells(split_context):
    wells, _, part = split_context
    dev = part.development_df
    cal = dev.iloc[part.calibration.cp_calib_idx]
    assert len(cal) == 178
    assert len(part.calibration.cv_pool_idx) == 712
    quotas = calibration_cell_quotas()
    for well in wells.training_well_names:
        for cls, expected in quotas[well].items():
            observed = int(
                ((cal["well_id"] == well) & (cal["quality_class_id"] == cls)).sum()
            )
            assert observed == expected
    assert cal["quality_class_id"].value_counts().sort_index().to_dict() == {
        0: 51,
        1: 49,
        2: 78,
    }


def test_authoritative_calibration_ids_roundtrip(split_context):
    _, _, part = split_context
    dev = part.development_df
    ids = dev.iloc[part.calibration.cp_calib_idx]["sample_id"].astype(str).tolist()
    rebuilt = make_calibration_split_from_sample_ids(
        dev, dev["quality_class_id"].to_numpy(dtype=int), ids
    )
    assert np.array_equal(rebuilt.cp_calib_idx, part.calibration.cp_calib_idx)
    assert np.array_equal(rebuilt.cv_pool_idx, part.calibration.cv_pool_idx)


def test_outer_fold_order_and_counts(split_context):
    _, _, part = split_context
    assert [f.validation_well for f in part.folds] == [
        "WF1",
        "CS12",
        "CS602",
        "CS607",
        "CS11",
    ]
    assert [len(f.val_idx) for f in part.folds] == [128, 114, 206, 125, 139]
    assert [len(f.train_idx) for f in part.folds] == [584, 598, 506, 587, 573]
    pool = set(part.calibration.cv_pool_idx.tolist())
    for fold in part.folds:
        assert set(fold.train_idx.tolist()).isdisjoint(fold.val_idx.tolist())
        assert set(fold.train_idx.tolist()) | set(fold.val_idx.tolist()) == pool
        assert fold.validation_well not in set(
            part.development_df.iloc[fold.train_idx]["well_id"]
        )


def test_outer_internal_early_stop_counts(split_context):
    _, _, part = split_context
    y = part.development_df["quality_class_id"].to_numpy(dtype=int)
    splits = [
        make_outer_early_stopping_split(part.development_df, y, fold, seed=20260827)
        for fold in part.folds
    ]
    assert [len(s.early_stop_idx) for s in splits] == [88, 90, 76, 89, 86]
    assert [len(s.gradient_idx) for s in splits] == [496, 508, 430, 498, 487]
    for fold, split in zip(part.folds, splits, strict=True):
        assert np.intersect1d(split.gradient_idx, split.early_stop_idx).size == 0
        assert len(split.gradient_idx) + len(split.early_stop_idx) == len(fold.train_idx)


def test_final_605_107_split(split_context):
    _, _, part = split_context
    y712 = part.development_df.iloc[part.calibration.cv_pool_idx][
        "quality_class_id"
    ].to_numpy(dtype=int)
    s = make_final_early_stopping_split(y712, seed=20260827)
    assert len(s.gradient_idx) == 605
    assert len(s.early_stop_idx) == 107
    assert np.intersect1d(s.gradient_idx, s.early_stop_idx).size == 0
