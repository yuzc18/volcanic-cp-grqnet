import numpy as np
import pandas as pd
import pytest

from src.data.io import match_core_to_continuous
from src.data.labels import FZIBoundaries, FZILabeler, assign_from_fzi, compute_fzi, label_stability_score
from src.features.group_a import RawLogClipper
from src.features.group_b import ERFLithologyClassifier, GROUP_B_COLUMNS
from src.features.group_c import GroupCRegressors
from src.features.group_d import GROUP_D_COLUMNS, build_group_d_continuous, match_group_d_to_core
from src.features.uft import UFTPreprocessor


def test_fzi_formula_and_boundary_ties():
    fzi = compute_fzi(np.array([5.0]), np.array([0.1]))
    phi = 0.05
    expected = 0.0314 * np.sqrt(0.1 / phi) / (phi / (1 - phi))
    assert np.allclose(fzi, expected)
    b = FZIBoundaries(0.7, 1.05)
    got = assign_from_fzi(np.array([0.69, 0.7, 1.049, 1.05, 1.2]), b)
    assert got.tolist() == [2, 1, 1, 0, 0]


def test_fzi_uses_log10_and_stability_single_start():
    por = np.array([4, 4.2, 4.5, 5, 6, 7, 8, 9, 10], dtype=float)
    perm = np.array([.01, .02, .03, .05, .1, .2, .5, 1, 2], dtype=float)
    lab = FZILabeler(n_init=10, random_state=42).fit(por, perm)
    assert np.allclose(lab.fit_.log10_fzi, np.log10(lab.fit_.fzi))
    s = label_stability_score(por, perm, lab.fit_.labels, n_repeats=5, base_seed=1)
    assert s.shape == (9,)
    assert ((s >= 0) & (s <= 1)).all()


def _continuous_example():
    depth = np.arange(0, 0.875, 0.125)
    vals = np.arange(len(depth), dtype=float)
    return pd.DataFrame({
        "log_id": [f"L{i}" for i in range(len(depth))],
        "well_id": "W1",
        "depth_m": depth,
        "GR": vals,
        "CNL": vals + 10,
        "DEN": vals + 20,
        "AC": vals + 30,
        "RLA5": vals + 40,
    })


def test_group_d_centered_and_causal_definitions_ddof1():
    logs = _continuous_example()
    clipper = RawLogClipper().fit(logs)
    centered = build_group_d_continuous(logs, clipper=clipper, causal=False)
    causal = build_group_d_continuous(logs, clipper=clipper, causal=True)
    i = 3
    assert centered.loc[i, "GR_prev1"] == 2
    assert centered.loc[i, "GR_next1"] == 4
    assert centered.loc[i, "GR_mean3"] == 3
    assert centered.loc[i, "GR_std3"] == pytest.approx(1.0)  # ddof=1 of [2,3,4]
    assert centered.loc[i, "GR_diff1"] == 1
    assert causal.loc[i, "GR_next1"] == 1  # x(i-2)
    assert causal.loc[i, "GR_mean3"] == 2  # [1,2,3]
    assert causal.loc[i, "GR_std3"] == pytest.approx(1.0)


def test_group_d_no_padding_for_edge_core_sample():
    logs = _continuous_example()
    clipper = RawLogClipper().fit(logs)
    d = build_group_d_continuous(logs, clipper=clipper)
    core = pd.DataFrame({
        "sample_id": ["S0"], "well_id": ["W1"], "depth_m": [0.0],
        "porosity_pct": [5.0], "permeability_mD": [0.1]
    })
    matched = match_core_to_continuous(core, logs)
    with pytest.raises(ValueError, match="no padding"):
        match_group_d_to_core(matched, d)


def test_erf_outputs_18_probabilities():
    rng = np.random.default_rng(4)
    n = 72
    y = np.repeat(np.arange(3), n // 3)
    X = rng.normal(size=(n, 5)) + y[:, None] * 0.7
    corpus = pd.DataFrame(X, columns=["GR", "CNL", "DEN", "AC", "RLA5"])
    corpus["lithology_class_id"] = y
    clf = ERFLithologyClassifier(n_estimators=3, random_state=2, max_depth=5).fit(corpus)
    p = clf.predict_frame(corpus.iloc[:8])
    assert list(p.columns) == GROUP_B_COLUMNS
    assert p.shape == (8, 18)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_group_c_and_uft_shape():
    rng = np.random.default_rng(10)
    n = 80
    logs = pd.DataFrame(rng.normal(size=(n, 5)), columns=["GR", "CNL", "DEN", "AC", "RLA5"])
    logs["porosity_pct"] = 6 + 0.4 * logs["CNL"] + rng.normal(0, .1, n)
    logs["permeability_mD"] = 10 ** (-1 + 0.1 * logs["GR"] + rng.normal(0, .05, n))
    c_model = GroupCRegressors(random_state=1, n_estimators=5).fit(logs.iloc[:60])
    C = c_model.predict(logs)
    assert C.as_array().shape == (n, 2)
    assert np.isfinite(C.predicted_log10_permeability).all()

    B = np.zeros((n, 18)); B[:, 0] = 1.0
    D = np.tile(np.arange(25, dtype=float), (n, 1)) + rng.normal(0, .01, (n,25))
    clipper = RawLogClipper().fit(logs.iloc[:60])
    pre = UFTPreprocessor().fit_on(logs.iloc[:60], B[:60], C.as_array()[:60], D[:60], raw_clipper=clipper)
    X = pre.transform(logs, B, C, D)
    assert X.shape == (n, 50)
    assert np.allclose(X[:, 5:23].sum(axis=1), 1.0)
