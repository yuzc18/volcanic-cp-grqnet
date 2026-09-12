from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.generate_synthetic_data import write_synthetic_inputs
from src.pipeline import build_context, context_with_calibration_sample_ids
from src.uncertainty.calibration_designs import (
    contiguous_blocks_design,
    repeat_seeds,
    stratified_within_well_design,
    whole_well_design,
)
from src.uncertainty.workflow import apply_randomized_aps


def _context(tmp_path: Path):
    d = tmp_path / "synthetic"
    write_synthetic_inputs(d, 20260827)
    return build_context(d, cp_seed=20260827)


def test_repeat_seed_policy_is_fixed_and_deterministic():
    assert repeat_seeds(20260827, 3) == (20260827, 20260828, 20260829)


def test_stratified_designs_hit_all_published_total_sizes(tmp_path):
    ctx = _context(tmp_path)
    wells = tuple(ctx.wells.training_well_names)
    blind = tuple(ctx.wells.blind_well_names)
    for frac, n in [(0.10, 89), (0.15, 134), (0.20, 178), (0.25, 223)]:
        d = stratified_within_well_design(
            ctx.development,
            n_cal=n,
            seed=20260827,
            well_order=wells,
            evaluation_wells=blind,
            name=f"x{n}",
            fraction=frac,
        )
        assert d.n_cal == n
        assert len(set(d.calibration_sample_ids)) == n


def test_contiguous_design_is_one_consecutive_sample_block_per_well(tmp_path):
    ctx = _context(tmp_path)
    wells = tuple(ctx.wells.training_well_names)
    d = contiguous_blocks_design(
        ctx.development,
        n_cal=178,
        seed=20260827,
        well_order=wells,
        evaluation_wells=tuple(ctx.wells.blind_well_names),
        name="blocks",
        fraction=0.20,
    )
    selected = set(d.calibration_sample_ids)
    assert len(selected) == 178
    for well in wells:
        g = ctx.development.loc[ctx.development["well_id"] == well].sort_values("depth_m")
        pos = np.flatnonzero(g["sample_id"].astype(str).isin(selected).to_numpy())
        assert len(pos) > 0
        assert np.array_equal(pos, np.arange(pos[0], pos[-1] + 1))


def test_whole_well_design_removes_entire_calibration_well_from_modeling(tmp_path):
    ctx = _context(tmp_path)
    d = whole_well_design(
        ctx.development,
        calibration_well="CS12",
        evaluation_wells=tuple(ctx.wells.blind_well_names),
        seed=20260827,
        name="whole",
    )
    assert d.n_cal == 142
    alt = context_with_calibration_sample_ids(
        ctx,
        list(d.calibration_sample_ids),
        extra_upstream_excluded_wells=d.whole_well_calibration,
    )
    assert set(alt.calibration_rows["well_id"].astype(str)) == {"CS12"}
    assert "CS12" not in set(alt.modeling_rows["well_id"].astype(str))
    assert alt.extra_upstream_excluded_wells == ("CS12",)


def test_workflow_uses_randomized_boundary_and_nonempty_sets():
    import pandas as pd

    cal = pd.DataFrame({
        "sample_id": ["c1", "c2", "c3", "c4"],
        "well_id": ["W"] * 4,
        "quality_class_id": [0, 1, 0, 2],
        "prob_High": [0.8, 0.2, 0.7, 0.1],
        "prob_Medium": [0.1, 0.7, 0.2, 0.2],
        "prob_Low": [0.1, 0.1, 0.1, 0.7],
    })
    ev = pd.DataFrame({
        "sample_id": ["e1", "e2"],
        "well_id": ["T", "T"],
        "quality_class_id": [0, 1],
        "prob_High": [0.8, 0.34],
        "prob_Medium": [0.1, 0.33],
        "prob_Low": [0.1, 0.33],
    })
    app = apply_randomized_aps(cal, ev, alpha=0.2, rng=np.random.default_rng(7))
    assert np.all(app.frame["cp_set_size"].to_numpy() >= 1)
    assert np.all(app.frame["cp_set_size"].to_numpy() <= 3)
    assert "cp_boundary_removed" in app.frame
    assert np.isfinite(app.q_hat)
