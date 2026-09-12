from __future__ import annotations

import numpy as np

from src.uncertainty.conformal import (
    ConformalAPS,
    aps_nonconformity_scores,
    build_prediction_sets,
    build_prediction_sets_detailed,
    compute_threshold,
    finite_sample_order_statistic,
)


def test_randomized_score_hand_example():
    p = np.array([[0.6, 0.3, 0.1]])
    y = np.array([1])
    s = aps_nonconformity_scores(p, y, uniforms=np.array([0.5]))
    assert abs(s[0] - 0.75) < 1e-12


def test_primary_order_statistic_is_171():
    assert finite_sample_order_statistic(178, 0.05) == 171


def test_threshold_is_exact_kth_order_not_numpy_quantile_surrogate():
    scores = np.arange(1, 179, dtype=float) / 1000
    assert compute_threshold(scores, 0.05) == scores[170]


def test_randomized_boundary_removal_formula():
    # q=0.75; sorted probabilities 0.6,0.3,0.1. Boundary is class 1 with
    # removal probability (0.9-0.75)/0.3 = 0.5.
    p = np.array([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
    out = build_prediction_sets_detailed(
        p, 0.75, uniforms=np.array([0.25, 0.75]), randomized=True
    )
    assert out.sets[0] == [0]       # removed because u < 0.5
    assert out.sets[1] == [0, 1]    # retained because u >= 0.5


def test_nonempty_guard_always_retains_top_class():
    # Boundary is rank 1; q is far below p_top, so randomized removal would
    # otherwise empty the set. The manuscript guard must keep the top class.
    p = np.array([[0.8, 0.15, 0.05]])
    sets, sizes = build_prediction_sets(
        p, 0.1, uniforms=np.array([0.0]), randomized=True, always_retain_top=True
    )
    assert sets == [[0]]
    assert sizes.tolist() == [1]


def _problem(n: int, seed: int):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 3))
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in p])
    return p, y


def test_randomized_nonempty_aps_has_nominal_marginal_coverage_on_exchangeable_data():
    coverages = []
    for trial in range(80):
        pc, yc = _problem(500, 1000 + trial)
        pt, yt = _problem(500, 5000 + trial)
        cp = ConformalAPS(alpha=0.05, seed=20260827 + trial)
        cp.calibrate(pc, yc)
        coverages.append(cp.evaluate(pt, yt).marginal_coverage)
    assert float(np.mean(coverages)) >= 0.94


def test_same_seed_same_single_use_randomization():
    pc, yc = _problem(200, 10)
    pt, _ = _problem(100, 20)
    cp1 = ConformalAPS(alpha=0.05, seed=20260827).calibrate(pc, yc)
    cp2 = ConformalAPS(alpha=0.05, seed=20260827).calibrate(pc, yc)
    s1, z1 = cp1.predict(pt)
    s2, z2 = cp2.predict(pt)
    assert cp1.q_hat == cp2.q_hat
    assert s1 == s2
    assert np.array_equal(z1, z2)
