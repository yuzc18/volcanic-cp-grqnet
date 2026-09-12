from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from src.analysis.ablation import TABLE5_CONFIGS, grq_ablation_inputs_and_config, select_groups
from src.analysis.capacity import capacity_scan_configs
from src.models.baselines import build_neural_baseline
from src.models.grq_net import GRQNet
from src.training.baseline_non_neural import (
    UnrecoveredSearchSpaceError,
    fixed_reported_params,
    load_search_protocol,
    tune_non_neural,
)


def _cfg():
    with open('configs/baselines.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_reported_neural_parameter_counts_exact():
    cfg = _cfg()
    for name in ('mlp', 'lstm', 'transformer'):
        build = build_neural_baseline(name, cfg)
        assert build.actual_parameters == build.expected_parameters


def test_tabnet_is_lazy_and_has_clear_dependency_error_or_builds():
    cfg = _cfg()
    try:
        build = build_neural_baseline('tabnet', cfg)
    except ImportError as exc:
        assert 'pytorch-tabnet==4.1.0' in str(exc)
    else:
        # Table 3(a) reports 108,728 trainable parameters for TabNet.
        assert build.actual_parameters == build.expected_parameters


def test_table5_feature_dimensions():
    X = np.zeros((2, 50), dtype=np.float32)
    dims = {name: select_groups(X, groups).shape[1] for name, groups in TABLE5_CONFIGS.items()}
    assert dims == {
        'A': 5,
        'A+B': 23,
        'A+B+C': 25,
        'A+B+D': 48,
        'A+C+D': 32,
        'A+B+C+D': 50,
    }


def test_grq_ablation_reinstantiates_active_groups_only():
    X = np.zeros((3, 50), dtype=np.float32)
    groups = TABLE5_CONFIGS['A+B']
    Xin, cfg = grq_ablation_inputs_and_config(X, groups)
    assert Xin.shape == (3, 23)
    assert cfg.group_dims == (5, 18)
    model = GRQNet(cfg)
    assert model(torch.zeros(3, 23)).shape == (3, 3)


def test_capacity_scan_has_exactly_seven_one_factor_variants():
    rows = capacity_scan_configs()
    assert len(rows) == 7
    assert [name for name, _ in rows] == [
        'reported', 'L=2', 'L=6', 'd_g=32', 'd_g=64', 'dropout=0', 'dropout=0.30'
    ]


def test_manuscript_hpo_spaces_are_complete_and_grid_sizes_match():
    protocol = load_search_protocol('configs/search_spaces.yaml')
    entries = protocol['search_protocol']
    for name in ('lightgbm', 'xgboost', 'catboost', 'multinomial_logistic_regression', 'random_forest'):
        assert entries[name].get('available') is True
        assert entries[name].get('space')
    assert len(entries['multinomial_logistic_regression']['space']) == 4
    assert len(entries['random_forest']['space']) == 18
    assert entries['lightgbm']['trials_per_outer_fold'] == 50
    assert entries['xgboost']['trials_per_outer_fold'] == 50
    assert entries['catboost']['trials_per_outer_fold'] == 50


def test_actual_logistic_grid_tuning_runs_from_release_config():
    protocol = load_search_protocol('configs/search_spaces.yaml')
    rng = np.random.default_rng(1)
    X = rng.normal(size=(75, 10))
    y = np.tile(np.arange(3), 25)
    r = tune_non_neural(
        'multinomial_logistic_regression',
        X[:45], y[:45], X[45:60], y[45:60],
        protocol=protocol, baseline_cfg=_cfg(), seed=1,
        refit_X=X[:60], refit_y=y[:60], smoke=True,
    )
    assert r.engine == 'grid'
    assert r.best_params['C'] in {0.01, 0.1, 1.0, 10.0}


def test_grid_tuning_engine_with_injected_toy_space():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(90, 8))
    y = np.tile(np.arange(3), 30)
    protocol = {
        'search_protocol': {
            'multinomial_logistic_regression': {
                'engine': 'grid',
                'n_configurations': 4,
                'numerical_space_recovered': True,
                'space': [
                    {'C': 0.01, 'penalty': 'l2', 'class_weight': 'balanced'},
                    {'C': 0.1, 'penalty': 'l2', 'class_weight': 'balanced'},
                    {'C': 1.0, 'penalty': 'l2', 'class_weight': 'balanced'},
                    {'C': 10.0, 'penalty': 'l2', 'class_weight': 'balanced'},
                ],
            }
        }
    }
    r = tune_non_neural(
        'multinomial_logistic_regression',
        X[:60], y[:60], X[60:75], y[60:75],
        protocol=protocol, baseline_cfg=_cfg(), seed=2,
        refit_X=X[:75], refit_y=y[:75],
    )
    assert r.engine == 'grid'
    assert 'C' in r.best_params
    assert np.asarray(r.model.predict(X[75:])).reshape(-1).shape == (15,)


def test_optuna_tuning_engine_with_injected_toy_space():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(90, 8))
    y = np.tile(np.arange(3), 30)
    protocol = {
        'search_protocol': {
            'lightgbm': {
                'engine': 'optuna',
                'trials_per_outer_fold': 2,
                'numerical_space_recovered': True,
                'space': [
                    {'name': 'n_estimators', 'type': 'categorical', 'choices': [5, 8]},
                    {'name': 'max_depth', 'type': 'categorical', 'choices': [2, 3]},
                    {'name': 'learning_rate', 'type': 'categorical', 'choices': [0.05]},
                    {'name': 'bagging_freq', 'type': 'categorical', 'choices': [0]},
                ],
            }
        }
    }
    r = tune_non_neural(
        'lightgbm',
        X[:60], y[:60], X[60:75], y[60:75],
        protocol=protocol, baseline_cfg=_cfg(), seed=3,
        refit_X=X[:75], refit_y=y[:75],
    )
    assert r.engine == 'optuna'
    assert set(r.best_params) == {'n_estimators', 'max_depth', 'learning_rate', 'bagging_freq'}
    assert np.asarray(r.model.predict(X[75:])).reshape(-1).shape == (15,)
