"""GRQ-Net interpretability analyses for Figure 10.

The manuscript-facing SHAP protocol is fold-wise DeepExplainer on model logits,
with 100 background rows sampled from that fold's training subset.  Feature
importance is mean absolute SHAP over validation samples and the three class
outputs; group-level SHAP is the arithmetic mean of feature-level values within
each UFT group.

DeepExplainer attributions are only meaningful if they are additive: for each
sample and output, the SHAP values plus the explainer's expected value must
reproduce the model logit.  With the pinned stack (SHAP 0.45.1, PyTorch 2.3.1)
GRQ-Net does **not** satisfy that property, for two independent reasons that
were isolated on minimal control models:

======================================  ==============  ======
Control model                           max residual    verdict
======================================  ==============  ======
``Linear``                              3.0e-07         pass
``Linear -> nn.ReLU -> Linear``         1.1e-07         pass
``Linear -> nn.GELU -> Linear``         2.9e-01         fail
``Linear -> nn.LayerNorm -> Linear``    2.1e-01         fail
``Linear -> F.relu -> Linear``          3.8e-01         fail
======================================  ==============  ======

1. ``nn.GELU`` and ``nn.LayerNorm`` have no DeepLIFT rule in SHAP 0.45.1; this
   is the source of its ``unrecognized nn.Module`` warnings.  GRQ-Net uses both
   in every group encoder, gated residual block and classification head.
2. DeepExplainer installs backward hooks on ``nn.Module`` instances, so
   operators called functionally are not attributed at all.  The last row above
   shows an ``F.relu`` network failing where the identical ``nn.ReLU`` network
   passes.  ``FullContextGroupAttention`` and ``GatedResidualBlock`` call
   ``F.relu`` and ``torch.sigmoid`` functionally.

Both causes must be addressed before DeepExplainer output can be reported for
this architecture: exposing the activations as modules fixes (2) but not (1).

:func:`compute_foldwise_shap` therefore always measures the additivity residual
itself and, by default, refuses to return attributions that fail it.  A
completed run is not on its own evidence that an explanation is valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.features.uft import UFT_COLUMNS, UFT_LAYOUT

GROUP_NAMES = ["A", "B", "C", "D"]
GROUP_SLICES = [UFT_LAYOUT.slice_a, UFT_LAYOUT.slice_b, UFT_LAYOUT.slice_c, UFT_LAYOUT.slice_d]


class ShapAdditivityError(RuntimeError):
    """Raised when DeepExplainer attributions fail the additivity check."""


@dataclass(frozen=True)
class AdditivityDiagnostic:
    """Per-fold additivity evidence for one SHAP explanation."""

    fold_id: int
    validation_well: str
    max_abs_residual: float
    max_abs_logit_delta: float
    tolerance: float
    n_background: int
    n_evaluated: int

    @property
    def passed(self) -> bool:
        return bool(self.max_abs_residual <= self.tolerance)

    def to_dict(self) -> dict:
        return {
            "fold": self.fold_id,
            "validation_well": self.validation_well,
            "max_abs_additivity_residual": self.max_abs_residual,
            "max_abs_logit_delta": self.max_abs_logit_delta,
            "tolerance": self.tolerance,
            "n_background": self.n_background,
            "n_evaluated": self.n_evaluated,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ShapSummary:
    feature_importance: pd.DataFrame
    group_importance: pd.DataFrame
    group_weight_summary: pd.DataFrame
    additivity: tuple[AdditivityDiagnostic, ...] = ()
    explainer: str = "shap.DeepExplainer"
    shap_version: str = ""

    @property
    def additivity_passed(self) -> bool:
        """True only if every fold satisfied the additivity check."""
        return bool(self.additivity) and all(d.passed for d in self.additivity)

    def additivity_frame(self) -> pd.DataFrame:
        return pd.DataFrame([d.to_dict() for d in self.additivity])


def _normalise_shap_array(values, n_samples: int, n_features: int, n_outputs: int) -> np.ndarray:
    """Return SHAP values as (sample, feature, output) across SHAP versions."""
    if isinstance(values, list):
        # Historical SHAP: one (sample, feature) array per output.
        arr = np.stack([np.asarray(v) for v in values], axis=-1)
    else:
        arr = np.asarray(values)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected DeepExplainer SHAP shape {arr.shape!r}.")
    if arr.shape == (n_samples, n_features, n_outputs):
        return arr
    if arr.shape == (n_outputs, n_samples, n_features):
        return np.moveaxis(arr, 0, -1)
    if arr.shape == (n_samples, n_outputs, n_features):
        return np.moveaxis(arr, 1, -1)
    raise ValueError(
        f"Cannot map DeepExplainer output shape {arr.shape!r} to "
        f"({n_samples}, {n_features}, {n_outputs})."
    )


def compute_foldwise_shap(
    fold_runs,
    *,
    background_size: int = 100,
    seed: int = 20260827,
    max_eval_rows_per_fold: int | None = None,
    additivity_tolerance: float = 0.01,
    on_additivity_failure: str = "error",
) -> ShapSummary:
    """Compute Figure-10 SHAP and group-weight summaries from trained outer folds.

    Parameters
    ----------
    additivity_tolerance
        Maximum absolute difference allowed between the SHAP values plus the
        explainer's expected value and the model logits.  SHAP's own internal
        tolerance is 0.01.
    on_additivity_failure
        ``"error"`` (default) raises :class:`ShapAdditivityError` if any fold
        exceeds the tolerance, so failed attributions are never written out as
        if they were valid.  ``"record"`` returns the attributions with the
        measured residuals attached and ``ShapSummary.additivity_passed`` False;
        callers that use it must label the output as diagnostically unverified.
    """
    if on_additivity_failure not in {"error", "record"}:
        raise ValueError("on_additivity_failure must be 'error' or 'record'.")
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise RuntimeError("SHAP is required; install requirements.txt.") from exc

    rng = np.random.default_rng(seed)
    diagnostics: list[AdditivityDiagnostic] = []
    weighted_feature_sum = np.zeros(len(UFT_COLUMNS), dtype=float)
    total_eval = 0
    weight_frames: list[pd.DataFrame] = []

    for run in fold_runs:
        X_train = np.asarray(run.prepared.X_outer_train, dtype=np.float32)
        X_eval = np.asarray(run.prepared.X_outer_val, dtype=np.float32)
        if max_eval_rows_per_fold is not None:
            X_eval = X_eval[: int(max_eval_rows_per_fold)]
        n_bg = min(int(background_size), len(X_train))
        if n_bg <= 0 or len(X_eval) <= 0:
            raise ValueError("SHAP requires non-empty train and validation matrices.")
        bg_idx = rng.choice(len(X_train), size=n_bg, replace=False)
        background = torch.from_numpy(X_train[bg_idx])
        eval_tensor = torch.from_numpy(X_eval)

        model = run.train_result.model
        model.eval()
        # GRQNet.forward returns raw logits by default, which is the manuscript
        # target for DeepExplainer.
        explainer = shap.DeepExplainer(model, background)
        # The check is run below against the model's own logits rather than
        # delegated to SHAP, so that the residual is measured and reported
        # instead of being silently suppressed.
        values = explainer.shap_values(eval_tensor, check_additivity=False)
        arr = _normalise_shap_array(values, len(X_eval), len(UFT_COLUMNS), 3)

        with torch.no_grad():
            logits = model(eval_tensor).cpu().numpy()
        expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
        reconstructed = arr.sum(axis=1) + expected[None, :]
        residual = np.abs(reconstructed - logits)
        diagnostics.append(
            AdditivityDiagnostic(
                fold_id=int(run.prepared.fold.fold_id),
                validation_well=str(run.prepared.fold.validation_well),
                max_abs_residual=float(residual.max()),
                max_abs_logit_delta=float(np.abs(logits - expected[None, :]).max()),
                tolerance=float(additivity_tolerance),
                n_background=int(n_bg),
                n_evaluated=int(len(X_eval)),
            )
        )
        fold_imp = np.abs(arr).mean(axis=(0, 2))
        weighted_feature_sum += fold_imp * len(X_eval)
        total_eval += len(X_eval)

        wf = run.oof_frame.iloc[: len(X_eval)][
            ["sample_id", "well_id", "quality_class_id", "alpha_A", "alpha_B", "alpha_C", "alpha_D"]
        ].copy()
        weight_frames.append(wf)

    failed = [d for d in diagnostics if not d.passed]
    if failed and on_additivity_failure == "error":
        worst = max(failed, key=lambda d: d.max_abs_residual)
        raise ShapAdditivityError(
            f"DeepExplainer attributions failed the additivity check on "
            f"{len(failed)} of {len(diagnostics)} folds; worst residual "
            f"{worst.max_abs_residual:.6g} on fold {worst.fold_id} "
            f"({worst.validation_well}) against tolerance {worst.tolerance:g}, "
            f"with a maximum absolute logit deviation of "
            f"{worst.max_abs_logit_delta:.6g} on the same rows. With the pinned "
            f"stack this is expected: SHAP 0.45.1 has no DeepLIFT rule for "
            f"nn.GELU or nn.LayerNorm, and its backward hooks do not see the "
            f"functional F.relu/torch.sigmoid calls inside the group-attention "
            f"and gated-residual blocks (see this module's docstring for the "
            f"isolating control models). Do not report these attributions as "
            f"Figure-10 evidence: use an explainer whose additivity can be "
            f"verified for this architecture, or pass "
            f"on_additivity_failure='record' to obtain the values together with "
            f"the failing diagnostic for inspection."
        )

    feature_imp = weighted_feature_sum / float(total_eval)
    feature_groups = (
        ["A"] * UFT_LAYOUT.dim_a
        + ["B"] * UFT_LAYOUT.dim_b
        + ["C"] * UFT_LAYOUT.dim_c
        + ["D"] * UFT_LAYOUT.dim_d
    )
    feature_frame = pd.DataFrame(
        {"feature": UFT_COLUMNS, "group": feature_groups, "mean_abs_shap": feature_imp}
    ).sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)

    group_rows = []
    for name, sl in zip(GROUP_NAMES, GROUP_SLICES):
        group_rows.append(
            {"group": name, "mean_feature_abs_shap": float(feature_imp[sl].mean())}
        )
    group_frame = pd.DataFrame(group_rows)

    weights = pd.concat(weight_frames, ignore_index=True)
    long = weights.melt(
        id_vars=["sample_id", "well_id", "quality_class_id"],
        value_vars=["alpha_A", "alpha_B", "alpha_C", "alpha_D"],
        var_name="group",
        value_name="group_weight",
    )
    long["group"] = long["group"].str.replace("alpha_", "", regex=False)
    weight_summary = (
        long.groupby(["quality_class_id", "group"], as_index=False)["group_weight"]
        .agg(["count", "median", "mean", "std"])
        .reset_index()
    )
    try:
        import shap as _shap
        version = str(getattr(_shap, "__version__", ""))
    except Exception:  # pragma: no cover - shap imported successfully above
        version = ""
    return ShapSummary(
        feature_frame,
        group_frame,
        weight_summary,
        additivity=tuple(diagnostics),
        explainer="shap.DeepExplainer",
        shap_version=version,
    )
