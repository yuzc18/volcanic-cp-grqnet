"""Non-neural baseline construction and tuning infrastructure.

The manuscript requires per-fold HPO for the five non-neural baselines.  The
release therefore ships machine-readable full search spaces in
``configs/search_spaces.yaml``.  The reported Table-3(a) final settings are
contained within those spaces, and every tuning run writes its selected
configuration for audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class UnrecoveredSearchSpaceError(RuntimeError):
    pass


@dataclass
class SelectionResult:
    model: Any
    best_params: dict[str, Any]
    internal_macro_f1: float
    engine: str


def _with_smoke_overrides(params: dict[str, Any], smoke: bool) -> dict[str, Any]:
    p = dict(params)
    if not smoke:
        return p
    for key in ("n_estimators", "iterations"):
        if key in p:
            p[key] = min(int(p[key]), 12)
    return p


def fixed_reported_params(name: str, baseline_cfg: dict, *, smoke: bool = False) -> dict[str, Any]:
    return _with_smoke_overrides(dict(baseline_cfg["non_neural_baselines"][name]["table3a_final"]), smoke)


def build_non_neural_estimator(name: str, params: dict[str, Any], *, seed: int, n_classes: int = 3):
    name = name.lower()
    p = dict(params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=1, verbosity=-1, **p)
    if name == "multinomial_logistic_regression":
        return LogisticRegression(
            solver="lbfgs", max_iter=5000,
            random_state=seed, **p
        )
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss",
            tree_method="hist", n_jobs=1, random_state=seed, **p
        )
    if name == "catboost":
        from catboost import CatBoostClassifier
        if p.pop("class_weights", None) == "balanced":
            p["auto_class_weights"] = "Balanced"
        return CatBoostClassifier(
            loss_function="MultiClass", verbose=False, random_seed=seed,
            thread_count=1, allow_writing_files=False, **p
        )
    raise KeyError(name)


def load_search_protocol(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_space(name: str, protocol: dict) -> dict:
    entry = protocol["search_protocol"][name]
    if not entry.get("available", entry.get("numerical_space_recovered", False)) or not entry.get("space"):
        raise UnrecoveredSearchSpaceError(
            f"Search space for {name} is not available in configs/search_spaces.yaml."
        )
    return entry


def _suggest(trial, spec: dict):
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(spec["name"], spec["choices"])
    if kind == "int":
        return trial.suggest_int(spec["name"], int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)), log=bool(spec.get("log", False)))
    if kind == "float":
        return trial.suggest_float(spec["name"], float(spec["low"]), float(spec["high"]), step=spec.get("step"), log=bool(spec.get("log", False)))
    raise ValueError(f"Unsupported search-space type: {kind}")


def tune_non_neural(
    name: str,
    X_gradient: np.ndarray,
    y_gradient: np.ndarray,
    X_internal: np.ndarray,
    y_internal: np.ndarray,
    *,
    protocol: dict,
    baseline_cfg: dict,
    seed: int,
    refit_X: np.ndarray | None = None,
    refit_y: np.ndarray | None = None,
    smoke: bool = False,
    n_classes: int = 3,
) -> SelectionResult:
    entry = _require_space(name, protocol)
    engine = entry["engine"]
    if engine == "grid":
        grid = entry["space"]
        expected = int(entry["n_configurations"])
        if len(grid) != expected:
            raise ValueError(f"{name} grid has {len(grid)} configurations, expected {expected}.")
        best_score, best_params = -1.0, None
        for params in grid:
            fit_params = _with_smoke_overrides(params, smoke)
            model = build_non_neural_estimator(name, fit_params, seed=seed, n_classes=n_classes)
            model.fit(X_gradient, y_gradient)
            score = float(f1_score(y_internal, np.asarray(model.predict(X_internal)).reshape(-1), average="macro"))
            if score > best_score:
                best_score, best_params = score, dict(params)
    elif engine == "optuna":
        import optuna
        specs = entry["space"]
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        def objective(trial):
            params = {spec["name"]: _suggest(trial, spec) for spec in specs}
            fit_params = _with_smoke_overrides(params, smoke)
            model = build_non_neural_estimator(name, fit_params, seed=seed, n_classes=n_classes)
            model.fit(X_gradient, y_gradient)
            return float(f1_score(y_internal, np.asarray(model.predict(X_internal)).reshape(-1), average="macro"))
        n_trials = 2 if smoke else int(entry["trials_per_outer_fold"])
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_score, best_params = float(study.best_value), dict(study.best_params)
    else:
        raise ValueError(engine)
    model = build_non_neural_estimator(
        name, _with_smoke_overrides(best_params, smoke), seed=seed, n_classes=n_classes
    )
    fitX = X_gradient if refit_X is None else refit_X
    fity = y_gradient if refit_y is None else refit_y
    model.fit(fitX, fity)
    return SelectionResult(model, best_params, best_score, engine)
