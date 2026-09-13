"""Regressions for the issues raised in the 2026-09-13 repository review.

Each test names the review item it guards and, where the review published a
probe result, reproduces that probe so the fixed behaviour is checked rather
than merely asserted in prose.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.analysis.interpretability import AdditivityDiagnostic, compute_foldwise_shap
from src.config import seed_before_model_construction
from src.models.baselines import build_neural_baseline
from src.pipeline import RuntimeOptions, prepare_final_model, prepare_outer_fold
from src.training.baseline_non_neural import load_search_protocol, tune_non_neural
from src.uncertainty.conformal import (
    InsufficientCalibrationError,
    build_prediction_sets_detailed,
    compute_threshold,
    finite_sample_order_statistic,
)
from src.uncertainty.workflow import (
    DEPLOYMENT_UNIT,
    apply_randomized_aps,
    conformal_substream,
    fold_unit,
)

ROOT = Path(__file__).resolve().parents[1]
PROB_COLS = ["prob_High", "prob_Medium", "prob_Low"]


def _frame(rng: np.random.Generator, n: int, n_classes: int = 3) -> pd.DataFrame:
    p = rng.dirichlet([2.0] * n_classes, size=n)
    out = {c: p[:, i] for i, c in enumerate(PROB_COLS[:n_classes])}
    out["quality_class_id"] = rng.integers(0, n_classes, n)
    out["sample_id"] = [f"s{i}" for i in range(n)]
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# C01 -- Table 8 selects LightGBM hyperparameters instead of reusing a fixed set
# --------------------------------------------------------------------------
def test_table8_gbm_path_calls_the_tuner_and_labels_the_fixed_branch():
    src = (ROOT / "scripts" / "run_label_sensitivity.py").read_text(encoding="utf-8")
    gbm = src[src.index("def _gbm_cv") : src.index("def main(")]
    assert "tune_non_neural(" in gbm, "default Table-8 path must run per-fold selection"
    assert "reported_table3a_fixed_demo_not_hpo" in gbm
    # The fixed-configuration branch must never be reported as HPO.
    assert "restored manuscript HPO" not in src
    assert "reported_fixed_demo" in gbm


def test_tuner_honours_the_class_count_of_the_label_definition():
    """Alternative Table-8 definitions use c=2 and c=4, so 3 must not be assumed."""
    protocol = load_search_protocol(ROOT / "configs" / "search_spaces.yaml")
    cfg = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    for n_classes in (2, 4):
        X = rng.normal(size=(80, 6))
        y = np.tile(np.arange(n_classes), 80 // n_classes)
        sel = tune_non_neural(
            "lightgbm", X[:60], y[:60], X[60:], y[60:],
            protocol=protocol, baseline_cfg=cfg, seed=7, smoke=True,
            n_classes=n_classes,
        )
        assert len(np.unique(sel.model.classes_)) == n_classes
        assert sel.engine == "optuna"
        assert sel.best_params, "a real selection must record the chosen parameters"


# --------------------------------------------------------------------------
# C02 -- the ARI population is the 890 development samples
# --------------------------------------------------------------------------
def test_ari_is_restricted_to_the_development_samples():
    src = (ROOT / "scripts" / "run_label_sensitivity.py").read_text(encoding="utf-8")
    assert "adjusted_rand_score(adopted[devmask], labels_all[devmask])" in src
    assert "adjusted_rand_score(adopted, labels_all)" not in src
    assert '"n_ari_samples"' in src, "the ARI population size must be exported"


# --------------------------------------------------------------------------
# C03 -- neural baselines are seeded before their weights are drawn
# --------------------------------------------------------------------------
def _initial_weight_hash(name: str, ambient_seed: int, requested_seed: int = 20260827) -> str:
    cfg = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text(encoding="utf-8"))
    torch.manual_seed(ambient_seed)
    np.random.seed(ambient_seed)
    torch.randn(37)
    np.random.rand(11)
    seed_before_model_construction(requested_seed)
    build = build_neural_baseline(name, cfg)
    h = hashlib.sha256()
    for _, tensor in sorted(build.model.state_dict().items()):
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("name", ["mlp", "lstm"])
def test_initial_weights_do_not_depend_on_the_ambient_random_state(name):
    """Review probe: ambient seeds 111 and 222 gave different weights before the fix."""
    assert _initial_weight_hash(name, 111) == _initial_weight_hash(name, 222)


@pytest.mark.parametrize("name", ["mlp", "lstm"])
def test_requested_seed_still_changes_the_initialization(name):
    a = _initial_weight_hash(name, 111, requested_seed=20260827)
    b = _initial_weight_hash(name, 111, requested_seed=20260828)
    assert a != b


def test_baseline_runner_seeds_before_constructing_the_model():
    src = inspect.getsource(
        __import__("src.training.baseline_runner", fromlist=["run_neural_on_prepared"]).run_neural_on_prepared
    )
    assert src.index("seed_before_model_construction") < src.index("build_neural_baseline")


# --------------------------------------------------------------------------
# C04 -- both blind-well CP entry points produce the same result
# --------------------------------------------------------------------------
def test_deployment_substream_does_not_depend_on_preceding_folds():
    """Review probe: q-hat differed (0.96249 vs 0.97764) and 93/369 sets disagreed."""
    rng = np.random.default_rng(3)
    cal, blind = _frame(rng, 178), _frame(rng, 369)
    seed = 20260827

    # conformal_calibrate.py: the deployment step follows five fold evaluations.
    for fold_id in (1, 2, 3, 4, 5):
        apply_randomized_aps(
            _frame(rng, 178), _frame(rng, 140),
            alpha=0.05, rng=conformal_substream(seed, fold_unit(fold_id)),
        )
    main = apply_randomized_aps(
        cal, blind, alpha=0.05, rng=conformal_substream(seed, DEPLOYMENT_UNIT)
    )
    # eval_blind.py: the deployment step alone.
    standalone = apply_randomized_aps(
        cal, blind, alpha=0.05, rng=conformal_substream(seed, DEPLOYMENT_UNIT)
    )

    assert main.q_hat == standalone.q_hat
    assert list(main.frame["cp_set"]) == list(standalone.frame["cp_set"])
    assert main.report.marginal_coverage == standalone.report.marginal_coverage


def test_substreams_are_distinct_stable_and_seed_dependent():
    a = conformal_substream(20260827, DEPLOYMENT_UNIT).uniform(size=5)
    b = conformal_substream(20260827, fold_unit(1)).uniform(size=5)
    c = conformal_substream(20260827, DEPLOYMENT_UNIT).uniform(size=5)
    d = conformal_substream(20260828, DEPLOYMENT_UNIT).uniform(size=5)
    assert not np.allclose(a, b)   # different units
    assert np.allclose(a, c)       # reproducible within a run and across runs
    assert not np.allclose(a, d)   # the seed still controls the stream


def test_eval_blind_entry_point_uses_the_shared_substream():
    src = (ROOT / "scripts" / "eval_blind.py").read_text(encoding="utf-8")
    assert "conformal_substream(args.seed, DEPLOYMENT_UNIT)" in src
    assert "np.random.default_rng(args.seed)" not in src


# --------------------------------------------------------------------------
# C05 -- SHAP additivity is measured, and failure is not silently accepted
# --------------------------------------------------------------------------
def test_shap_additivity_is_never_silently_disabled():
    src = (ROOT / "src" / "analysis" / "interpretability.py").read_text(encoding="utf-8")
    # The explainer call still passes check_additivity=False, but only because the
    # residual is computed here instead; the measurement must be present.
    assert "AdditivityDiagnostic(" in src
    assert "max_abs_residual" in src
    assert "ShapAdditivityError" in src


def test_additivity_diagnostic_threshold_semantics():
    kw = dict(fold_id=1, validation_well="WF1", max_abs_logit_delta=0.4,
              tolerance=0.01, n_background=100, n_evaluated=12)
    assert AdditivityDiagnostic(max_abs_residual=0.001, **kw).passed
    assert not AdditivityDiagnostic(max_abs_residual=0.199, **kw).passed


def test_invalid_additivity_policy_is_rejected():
    with pytest.raises(ValueError, match="on_additivity_failure"):
        compute_foldwise_shap([], on_additivity_failure="ignore")


def test_deep_explainer_additivity_control_models():
    """Documents why GRQ-Net fails: unsupported modules and functional ops."""
    shap = pytest.importorskip("shap")
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(60, 8)).astype(np.float32)
    bg, ev = torch.from_numpy(X[:50]), torch.from_numpy(X[50:])

    def max_residual(model):
        model.eval()
        explainer = shap.DeepExplainer(model, bg)
        values = np.asarray(explainer.shap_values(ev, check_additivity=False))
        expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
        with torch.no_grad():
            out = model(ev).numpy()
        return float(np.abs(values.sum(axis=1) + expected[None, :] - out).max())

    class Functional(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = torch.nn.Linear(8, 16), torch.nn.Linear(16, 3)

        def forward(self, x):
            return self.b(torch.relu(self.a(x)))

    module_relu = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    assert max_residual(module_relu) <= 0.01          # nn.ReLU is hooked
    assert max_residual(Functional()) > 0.01          # torch.relu is not
    gelu = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 3))
    assert max_residual(gelu) > 0.01                  # no DeepLIFT rule for GELU


# --------------------------------------------------------------------------
# C06 -- the Group D variant reaches training, not only the latency benchmark
# --------------------------------------------------------------------------
def test_runtime_options_expose_and_validate_the_group_d_variant():
    assert RuntimeOptions().group_d_variant == "centered"
    assert RuntimeOptions().group_d_causal is False
    assert RuntimeOptions(group_d_variant="causal").group_d_causal is True
    with pytest.raises(ValueError, match="group_d_variant"):
        RuntimeOptions(group_d_variant="trailing")


@pytest.mark.parametrize("fn", [prepare_outer_fold, prepare_final_model])
def test_training_preparation_uses_the_selected_variant(fn):
    src = inspect.getsource(fn)
    assert "causal=options.group_d_causal" in src
    assert "causal=False" not in src, "the variant must not be hard-coded"


def test_variant_comparison_driver_exists_and_covers_both_arms():
    path = ROOT / "scripts" / "run_group_d_variant_comparison.py"
    assert path.exists(), "P139 reports this comparison, so a driver must exist"
    src = path.read_text(encoding="utf-8")
    assert 'VARIANTS = ("centered", "causal")' in src
    assert "causal_minus_centered_pp" in src


# --------------------------------------------------------------------------
# C09 -- the n+1 order statistic is not silently clipped
# --------------------------------------------------------------------------
def test_manuscript_designs_are_unaffected():
    assert finite_sample_order_statistic(178, 0.05) == 171   # primary design
    assert finite_sample_order_statistic(89, 0.05) == 86     # smallest Table-7 design
    for alpha in (0.01, 0.10, 0.30):
        assert finite_sample_order_statistic(178, alpha) <= 178


def test_insufficient_calibration_is_reported_not_clipped():
    with pytest.raises(InsufficientCalibrationError) as excinfo:
        finite_sample_order_statistic(2, 0.05)
    assert excinfo.value.required_rank == 3
    with pytest.raises(InsufficientCalibrationError):
        compute_threshold(np.array([0.1, 0.4]), 0.05)


def test_conservative_full_label_branch_is_available():
    q = compute_threshold(np.array([0.1, 0.4]), 0.05, on_insufficient_calibration="full_set")
    assert np.isposinf(q)
    details = build_prediction_sets_detailed(np.array([[0.7, 0.2, 0.1]]), q)
    assert details.sizes.tolist() == [3]
    assert sorted(details.sets[0]) == [0, 1, 2]
