from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.analysis.benchmark import _causal_d_from_clipped_window
from src.analysis.conformal_analysis import (
    accuracy_by_set_size,
    coverage_curve_from_folds,
    verification_recovery_curves,
)
from src.analysis.interpretability import _normalise_shap_array
from src.analysis.figures import figure7_confusion_matrices, figure8_conformal_behavior
from src.analysis.statistics import (
    clopper_pearson_interval,
    greedy_thin_positions,
    moving_block_resample_positions,
    paired_t_interval,
)


def test_clopper_pearson_endpoints_and_paired_t():
    a = clopper_pearson_interval(0, 10)
    b = clopper_pearson_interval(10, 10)
    assert a.estimate == 0 and a.lower == 0 and 0 < a.upper < 1
    assert b.estimate == 1 and b.upper == 1 and 0 < b.lower < 1
    t = paired_t_interval([0.1, 0.1, 0.1, 0.1, 0.1])
    assert np.isclose(t.estimate, 0.1)
    assert np.isclose(t.lower, 0.1) and np.isclose(t.upper, 0.1)


def test_moving_block_bootstrap_preserves_per_well_counts():
    df = pd.DataFrame(
        {
            "well_id": ["A"] * 5 + ["B"] * 4,
            "depth_m": [0, 0.5, 1, 1.5, 2, 10, 10.5, 11, 11.5],
        }
    )
    idx = moving_block_resample_positions(df, rng=np.random.default_rng(1), block_length_m=1.0)
    sampled = df.iloc[idx]
    assert len(sampled) == len(df)
    assert sampled["well_id"].value_counts().to_dict() == {"A": 5, "B": 4}


def test_greedy_thinning_two_starting_choices():
    df = pd.DataFrame({"well_id": ["A"] * 6, "depth_m": [0.0, 0.2, 0.5, 0.7, 1.0, 1.2]})
    p0 = greedy_thin_positions(df, minimum_spacing_m=0.5, starting_choice=0)
    p1 = greedy_thin_positions(df, minimum_spacing_m=0.5, starting_choice=1)
    assert p0.tolist() == [0, 2, 4]
    assert p1.tolist() == [1, 3, 5]


def test_coverage_curve_reuses_fixed_predictions_and_is_30_points():
    # Uniform probabilities make the test independent of a particular fitted model.
    cal = pd.DataFrame(
        {
            "quality_class_id": [0, 1, 2, 0, 1, 2],
            "prob_High": [0.7, 0.2, 0.1, 0.6, 0.2, 0.2],
            "prob_Medium": [0.2, 0.7, 0.2, 0.3, 0.6, 0.2],
            "prob_Low": [0.1, 0.1, 0.7, 0.1, 0.2, 0.6],
        }
    )
    oof = cal.copy()
    run = SimpleNamespace(calibration_frame=cal, oof_frame=oof)
    curve = coverage_curve_from_folds([run] * 5, seed=20260827)
    assert len(curve.alpha) == 30
    assert np.isclose(curve.alpha[0], 0.01) and np.isclose(curve.alpha[-1], 0.30)
    assert np.all((curve.empirical_coverage >= 0) & (curve.empirical_coverage <= 1))


def test_verification_budget_uses_floor_count_at_ten_percent():
    n = 20
    df = pd.DataFrame(
        {
            "quality_class_id": np.zeros(n, dtype=int),
            "predicted_class_id": np.array([1, 1] + [0] * 18),
            "prob_High": np.linspace(0.1, 0.9, n),
            "prob_Medium": np.linspace(0.8, 0.05, n),
            "prob_Low": np.full(n, 0.1),
            "cp_set_size": np.array([3, 3] + [1] * 18),
        }
    )
    # At 10% of 20, exactly two rows are selected; CP ranking recovers both errors.
    c = verification_recovery_curves(df, budget_fraction=[0.10], random_draws=10, seed=1)
    assert np.isclose(c.cp_guided[0], 1.0)


def test_accuracy_by_set_size_computed_not_hardcoded():
    df = pd.DataFrame(
        {
            "cp_set_size": [1, 1, 2, 3],
            "predicted_class_id": [0, 1, 1, 2],
            "quality_class_id": [0, 0, 1, 1],
        }
    )
    out = accuracy_by_set_size(df).set_index("set_size")
    assert np.isclose(out.loc[1, "accuracy"], 0.5)
    assert np.isclose(out.loc[2, "accuracy"], 1.0)
    assert np.isclose(out.loc[3, "accuracy"], 0.0)


def test_causal_group_d_window_definition():
    clipped = np.array(
        [
            [1, 10, 100, 1000, 10000],
            [2, 20, 200, 2000, 20000],
            [4, 40, 400, 4000, 40000],
        ],
        dtype=float,
    )
    d = _causal_d_from_clipped_window(clipped)
    assert d.shape == (1, 25)
    # First curve slots: prev1=x(i-1), causal next-slot=x(i-2), mean, sample std, diff.
    assert np.isclose(d[0, 0], 2)
    assert np.isclose(d[0, 1], 1)
    assert np.isclose(d[0, 2], (1 + 2 + 4) / 3)
    assert np.isclose(d[0, 3], np.std([1, 2, 4], ddof=1))
    assert np.isclose(d[0, 4], 2)


def test_shap_array_normalizer_accepts_new_and_legacy_shapes():
    new = np.zeros((2, 50, 3))
    assert _normalise_shap_array(new, 2, 50, 3).shape == (2, 50, 3)
    legacy = [np.zeros((2, 50)) for _ in range(3)]
    assert _normalise_shap_array(legacy, 2, 50, 3).shape == (2, 50, 3)


def test_figure7_builder_accepts_six_model_oof_predictions(tmp_path):
    models = ["mlp", "transformer", "lightgbm", "tabnet", "catboost", "grqnet"]
    rows = []
    for model in models:
        for cls in range(3):
            rows.append({"model": model, "quality_class_id": cls, "predicted_class_id": cls})
    df = pd.DataFrame(rows)
    out = tmp_path / "fig7.png"
    figure7_confusion_matrices(df, output=out)
    assert out.exists() and out.stat().st_size > 0


def test_figure8_adds_trivial_one_one_reference_point(tmp_path):
    alpha = np.arange(0.01, 0.31, 0.01)
    nominal = 1.0 - alpha
    empirical = np.clip(nominal + 0.01, 0, 1)
    from src.analysis.conformal_analysis import CoverageCurve
    curve = CoverageCurve(alpha, nominal, empirical, 0.01)
    rows = []
    for cls in range(3):
        for i in range(4):
            rows.append({
                'quality_class_id': cls,
                'predicted_class_id': cls if i < 3 else (cls + 1) % 3,
                'cp_set_size': (i % 3) + 1,
            })
    fig = figure8_conformal_behavior(curve, pd.DataFrame(rows), output=tmp_path/'fig8.png')
    ax = fig.axes[0]
    pts = []
    for line in ax.lines:
        xs = np.asarray(line.get_xdata(), dtype=float)
        ys = np.asarray(line.get_ydata(), dtype=float)
        pts.extend(zip(xs.tolist(), ys.tolist()))
    assert any(np.isclose(x, 1.0) and np.isclose(y, 1.0) for x, y in pts)
