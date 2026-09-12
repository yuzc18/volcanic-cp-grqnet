#!/usr/bin/env python
"""Fast Round-4 component smoke test without proprietary data."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.analysis.ablation import TABLE5_CONFIGS, select_groups
from src.analysis.capacity import capacity_scan_configs
from src.models.baselines import build_neural_baseline
from src.training.baseline_neural import train_neural_baseline, predict_torch_classifier
from src.training.baseline_non_neural import build_non_neural_estimator, fixed_reported_params
from src.training.trainer import TrainConfig


def main():
    rng=np.random.default_rng(20260827)
    X=rng.normal(size=(120,50)).astype('float32'); y=np.tile(np.arange(3),40); rng.shuffle(y)
    tr=np.arange(90); ev=np.arange(90,120)
    with open('configs/baselines.yaml',encoding='utf-8') as f: cfg=yaml.safe_load(f)
    for name in ['mlp','lstm','transformer']:
        b=build_neural_baseline(name,cfg)
        assert b.actual_parameters==b.expected_parameters,(name,b.actual_parameters,b.expected_parameters)
        fit=train_neural_baseline(b.model,X[tr],y[tr],X[ev],y[ev],class_weight_labels=y[tr],train_cfg=TrainConfig(max_epochs=2,early_stop_patience=2),seed=20260827)
        pred,_=predict_torch_classifier(fit.model,X[ev]); assert pred.shape==(30,)
    for name in ['lightgbm','multinomial_logistic_regression','random_forest','xgboost','catboost']:
        p=fixed_reported_params(name,cfg,smoke=True); m=build_non_neural_estimator(name,p,seed=20260827); m.fit(X[tr],y[tr]); assert np.asarray(m.predict(X[ev])).reshape(-1).shape==(30,)
    assert [x[0] for x in capacity_scan_configs()]==['reported','L=2','L=6','d_g=32','d_g=64','dropout=0','dropout=0.30']
    assert {k:select_groups(X,v).shape[1] for k,v in TABLE5_CONFIGS.items()}=={'A':5,'A+B':23,'A+B+C':25,'A+B+D':48,'A+C+D':32,'A+B+C+D':50}
    print('Round-4 baseline smoke: PASS')

if __name__=='__main__': main()
