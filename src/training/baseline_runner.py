"""Outer-fold runners for Table 6(a) baseline comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.models.baselines import build_neural_baseline
from src.pipeline import PreparedOuterFold
from src.training.baseline_neural import evaluate_neural_baseline, train_neural_baseline
from src.training.baseline_non_neural import (
    SelectionResult,
    build_non_neural_estimator,
    fixed_reported_params,
    load_search_protocol,
    tune_non_neural,
)
from src.training.metrics import ClassificationMetrics, compute_metrics
from src.training.trainer import TrainConfig


@dataclass
class BaselineFoldResult:
    model_name: str
    fold_id: int
    validation_well: str
    metrics: ClassificationMetrics
    selected_params: dict[str, Any]
    selection_engine: str
    internal_macro_f1: float | None
    predictions: np.ndarray


def load_baseline_config(path: str | Path = "configs/baselines.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_neural_on_prepared(
    name: str,
    p: PreparedOuterFold,
    *,
    baseline_cfg: dict,
    train_cfg: TrainConfig,
    seed: int,
) -> BaselineFoldResult:
    build = build_neural_baseline(name, baseline_cfg, input_dim=p.X_outer_train.shape[1])
    # Exact parameter-count checks are manuscript-facing for MLP/LSTM/Transformer.
    # TabNet is checked when the pinned external package is available.
    if name in {"mlp", "lstm", "transformer"} and build.actual_parameters != build.expected_parameters:
        raise AssertionError(
            f"{name} parameter count {build.actual_parameters} != reported {build.expected_parameters}"
        )
    g, e = p.gradient_local_idx, p.early_stop_local_idx
    fit = train_neural_baseline(
        build.model,
        p.X_outer_train[g], p.y_outer_train[g],
        p.X_outer_train[e], p.y_outer_train[e],
        class_weight_labels=p.y_outer_train,
        train_cfg=train_cfg,
        seed=seed,
    )
    pred = evaluate_neural_baseline(fit, p.X_outer_val, p.y_outer_val)
    from src.training.baseline_neural import predict_torch_classifier
    yhat, _ = predict_torch_classifier(fit.model, p.X_outer_val)
    return BaselineFoldResult(
        name, p.fold.fold_id, p.fold.validation_well, pred,
        selected_params={}, selection_engine="fixed_architecture_shared_training",
        internal_macro_f1=float(fit.best_val_macro_f1), predictions=yhat,
    )


def run_non_neural_on_prepared(
    name: str,
    p: PreparedOuterFold,
    *,
    baseline_cfg: dict,
    search_protocol: dict,
    seed: int,
    mode: str = "manuscript_hpo",
    smoke: bool = False,
) -> BaselineFoldResult:
    g, e = p.gradient_local_idx, p.early_stop_local_idx
    if mode == "manuscript_hpo":
        sel = tune_non_neural(
            name,
            p.X_outer_train[g], p.y_outer_train[g],
            p.X_outer_train[e], p.y_outer_train[e],
            protocol=search_protocol,
            baseline_cfg=baseline_cfg,
            seed=seed,
            # Explicit implementation choice; see configs/baselines.yaml.
            refit_X=p.X_outer_train,
            refit_y=p.y_outer_train,
            smoke=smoke,
        )
        model = sel.model
        params = sel.best_params
        engine = sel.engine
        internal = sel.internal_macro_f1
    elif mode == "reported_fixed_demo":
        params = fixed_reported_params(name, baseline_cfg, smoke=smoke)
        model = build_non_neural_estimator(name, params, seed=seed)
        model.fit(p.X_outer_train, p.y_outer_train)
        engine = "reported_table3a_fixed_demo_not_hpo"
        internal = None
    else:
        raise ValueError(mode)
    yhat = np.asarray(model.predict(p.X_outer_val), dtype=int).reshape(-1)
    metrics = compute_metrics(p.y_outer_val, yhat)
    return BaselineFoldResult(
        name, p.fold.fold_id, p.fold.validation_well, metrics,
        selected_params=params, selection_engine=engine,
        internal_macro_f1=internal, predictions=yhat,
    )


@dataclass
class FinalNonNeuralResult:
    model_name: str
    metrics: ClassificationMetrics
    selected_params: dict[str, Any]
    selection_engine: str
    internal_macro_f1: float | None
    predictions: np.ndarray


def run_final_non_neural_prepared(
    name: str,
    p,
    *,
    baseline_cfg: dict,
    search_protocol: dict,
    seed: int,
    mode: str = "manuscript_hpo",
    smoke: bool = False,
) -> FinalNonNeuralResult:
    """Tune on the 712-row final modeling set and refit on all 712 rows.

    This full-712 refit is explicitly stated in Table 3(a)'s note.
    """
    g, e = p.gradient_local_idx, p.early_stop_local_idx
    if mode == "manuscript_hpo":
        sel = tune_non_neural(
            name,
            p.X_modeling[g], p.y_modeling[g],
            p.X_modeling[e], p.y_modeling[e],
            protocol=search_protocol,
            baseline_cfg=baseline_cfg,
            seed=seed,
            refit_X=p.X_modeling,
            refit_y=p.y_modeling,
            smoke=smoke,
        )
        model, params, engine, internal = sel.model, sel.best_params, sel.engine, sel.internal_macro_f1
    elif mode == "reported_fixed_demo":
        params = fixed_reported_params(name, baseline_cfg, smoke=smoke)
        model = build_non_neural_estimator(name, params, seed=seed)
        model.fit(p.X_modeling, p.y_modeling)
        engine, internal = "reported_table3a_fixed_demo_not_hpo", None
    else:
        raise ValueError(mode)
    yhat = np.asarray(model.predict(p.X_blind), dtype=int).reshape(-1)
    return FinalNonNeuralResult(
        name, compute_metrics(p.y_blind, yhat), params, engine, internal, yhat
    )
