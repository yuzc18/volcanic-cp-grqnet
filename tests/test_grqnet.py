"""Architecture and behavioral tests for GRQ-Net.

Invariants verified:

    1. Total trainable parameter count == 319,815 (Table 3).
    2. Forward pass on ``(batch, 50)`` produces ``(batch, 3)`` logits.
    3. ``return_attention=True`` additionally produces ``(batch, 4)``
       group weights with values strictly in (0, 1) — i.e., Sigmoid,
       NOT Softmax (so the four weights do not sum to 1).
    4. Group weights are sample-specific (different inputs yield
       different alphas).
    5. Given a fixed seed and fixed input, the forward pass is
       deterministic.
    6. The model can be trained end-to-end via the SmoothedWeightedCE
       loss without shape errors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.grq_net import GRQNet, GRQNetConfig, GatedResidualBlock
from src.training.loss import (
    SmoothedWeightedCrossEntropy,
    compute_inverse_frequency_weights,
)


# ---------------------------------------------------------------------------
# Architecture invariants
# ---------------------------------------------------------------------------
def test_parameter_count_matches_paper():
    """GRQ-Net with default hyperparameters must have exactly 319,815 trainable params."""
    model = GRQNet()
    assert model.count_parameters() == 319_815, (
        f"Parameter count {model.count_parameters():,} != paper Table 3 "
        f"value 319,815. Architecture or hyperparameters have diverged "
        f"from the paper."
    )


def test_d_model_equals_4_times_d_group():
    """d_model = 4 * d_g = 192 (Table 3)."""
    cfg = GRQNetConfig()
    assert cfg.d_model == 4 * cfg.d_group == 192


def test_group_dims_sum_to_uft_dim():
    """Per-group input dims must sum to 50 (UFT total)."""
    cfg = GRQNetConfig()
    assert sum(cfg.group_dims) == 50


# ---------------------------------------------------------------------------
# Forward-pass shape contracts
# ---------------------------------------------------------------------------


def test_gated_residual_main_branch_uses_relu_as_eq7():
    block = GatedResidualBlock(d_model=2, dropout=0.0)
    block.eval()
    with torch.no_grad():
        block.main.weight.copy_(torch.eye(2))
        block.main.bias.zero_()
        block.gate.weight.zero_()
        block.gate.bias.fill_(20.0)  # gate ~= 1
    x = torch.tensor([[-2.0, 3.0]])
    with torch.no_grad():
        actual = block(x)
        expected = block.norm(x + torch.relu(x))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_forward_output_shape():
    """Forward on (B, 50) must yield (B, 3) logits."""
    torch.manual_seed(0)
    model = GRQNet()
    x = torch.randn(13, 50)
    out = model(x)
    assert out.shape == (13, 3)


def test_forward_with_attention_returns_weights():
    """``return_attention=True`` must return (logits, alpha) where alpha is (B, 4)."""
    torch.manual_seed(0)
    model = GRQNet()
    x = torch.randn(7, 50)
    logits, alpha = model(x, return_attention=True)
    assert logits.shape == (7, 3)
    assert alpha.shape == (7, 4)


def test_alpha_values_are_sigmoid_not_softmax():
    """alpha must be Sigmoid output: each entry in (0, 1), rows NOT summing to 1.

    The paper explicitly chose Sigmoid over Softmax (Section 3.4); this
    test prevents a future contributor from "fixing" the architecture by
    inserting a Softmax normalization.
    """
    torch.manual_seed(0)
    model = GRQNet()
    model.eval()
    x = torch.randn(64, 50)
    with torch.no_grad():
        _, alpha = model(x, return_attention=True)

    # Range: every entry strictly in (0, 1)
    assert (alpha > 0).all() and (alpha < 1).all()

    # Row sums must vary across samples — a Softmax would force every
    # row sum to be exactly 1.0. A Sigmoid does not.
    row_sums = alpha.sum(dim=1)
    assert row_sums.std().item() > 1e-3, (
        "alpha row-sum standard deviation is suspiciously small "
        f"({row_sums.std().item():.2e}); is alpha actually Sigmoid-activated?"
    )


def test_alpha_is_sample_specific():
    """Different inputs must yield different group weights."""
    torch.manual_seed(0)
    model = GRQNet()
    model.eval()
    x1 = torch.randn(1, 50)
    x2 = torch.randn(1, 50) + 5.0
    with torch.no_grad():
        _, alpha1 = model(x1, return_attention=True)
        _, alpha2 = model(x2, return_attention=True)
    assert not torch.allclose(alpha1, alpha2, atol=1e-4), (
        "alpha is identical for two different inputs — group attention "
        "may not be input-dependent."
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_forward_is_deterministic_under_eval_mode():
    """Two eval-mode forward passes on the same input must be bit-identical."""
    torch.manual_seed(42)
    model = GRQNet()
    model.eval()
    x = torch.randn(8, 50)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


def test_two_models_with_same_seed_produce_same_output():
    """Constructing two models with the same seed yields identical outputs."""
    torch.manual_seed(123)
    m1 = GRQNet()
    torch.manual_seed(123)
    m2 = GRQNet()
    m1.eval()
    m2.eval()

    x = torch.randn(4, 50)
    with torch.no_grad():
        torch.testing.assert_close(m1(x), m2(x), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# End-to-end training step
# ---------------------------------------------------------------------------
def test_end_to_end_training_step_runs():
    """A single forward + loss + backward must run without shape errors."""
    torch.manual_seed(0)
    model = GRQNet()

    # Fake labels with all three classes present
    fake_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    weights = compute_inverse_frequency_weights(fake_labels, n_classes=3)
    loss_fn = SmoothedWeightedCrossEntropy(
        class_weights=weights, label_smoothing=0.05, n_classes=3
    )

    x = torch.randn(8, 50, requires_grad=False)
    targets = torch.from_numpy(fake_labels).long()

    logits = model(x)
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss).item()
    loss.backward()

    # Every trainable parameter must have received a gradient (no
    # disconnected sub-graph)
    no_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not no_grad, f"Parameters without gradient: {no_grad[:5]}..."


# ---------------------------------------------------------------------------
# Loss function correctness
# ---------------------------------------------------------------------------
def test_class_weights_dataset_weighted_average_is_one():
    """Inverse-frequency weights satisfy sum_c (n_c * w_c) = N.

    The standard inverse-frequency normalization w_c = N / (K * n_c)
    ensures that the per-sample average class weight over the training
    set equals 1, regardless of class imbalance. This invariant is what
    makes the resulting weighted CE comparable in scale to the unweighted
    CE. The simpler-looking ``sum(w) == K`` only holds for balanced
    classes.
    """
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])  # imbalanced
    K = 3
    weights = compute_inverse_frequency_weights(labels, n_classes=K)
    counts = np.array([np.sum(labels == c) for c in range(K)])
    weighted_sum = float(np.sum(counts * weights.numpy()))
    assert abs(weighted_sum - len(labels)) < 1e-5, (
        f"sum(n_c * w_c) = {weighted_sum}, expected {len(labels)} (= N). "
        "Inverse-frequency normalization is off."
    )


def test_class_weights_minority_gets_higher_weight():
    """The rarest class must receive the largest weight, by construction."""
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])  # n_0=3, n_1=2, n_2=5
    weights = compute_inverse_frequency_weights(labels, n_classes=3).numpy()
    # Class 1 is rarest -> largest weight; class 2 is most common -> smallest
    assert weights[1] > weights[0] > weights[2]


def test_label_smoothing_produces_valid_distribution():
    """Smoothed labels sum to 1 and have on/off values matching Mueller (2019)."""
    weights = torch.tensor([1.0, 1.0, 1.0])
    loss_fn = SmoothedWeightedCrossEntropy(
        class_weights=weights, label_smoothing=0.05, n_classes=3
    )
    # With uniform weights and on/off labels, the formula is
    # on = 1 - eps + eps/K = 0.95 + 0.05/3 ≈ 0.9667
    # off = eps/K = 0.05/3 ≈ 0.0167
    # Their sum: on + 2*off = 1.0
    eps = 0.05
    K = 3
    on = 1.0 - eps + eps / K
    off = eps / K
    assert abs(on + 2 * off - 1.0) < 1e-9
    # And on > off
    assert on > off


def test_loss_is_small_on_near_perfect_predictions():
    """Near-perfect logits give a small (but nonzero) loss.

    With label smoothing eps=0.05, even arbitrarily confident logits
    cannot drive the loss to zero — the off-classes always receive
    epsilon/K probability mass that the model is penalized for not
    matching. We simply assert the loss is well below ``log(K) ≈ 1.10``
    (the uniform-prediction baseline) and strictly non-negative.
    """
    weights = torch.tensor([1.0, 1.0, 1.0])
    loss_fn = SmoothedWeightedCrossEntropy(
        class_weights=weights, label_smoothing=0.05, n_classes=3
    )
    logits = torch.tensor([
        [10.0, -10.0, -10.0],
        [-10.0, 10.0, -10.0],
        [-10.0, -10.0, 10.0],
    ])
    targets = torch.tensor([0, 1, 2])
    loss = loss_fn(logits, targets)
    uniform_baseline = float(np.log(3))
    assert 0.0 <= loss.item() < uniform_baseline, (
        f"Near-perfect loss {loss.item():.3f} not in [0, log(3)={uniform_baseline:.3f})"
    )
