#!/usr/bin/env python
"""Standalone six-stage latency benchmark matching Section 3.6."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.generate_synthetic_data import write_synthetic_inputs
from src.analysis.benchmark import benchmark_end_to_end, prepare_latency_assets
from src.pipeline import RuntimeOptions, run_main_experiment
from src.uncertainty.workflow import run_main_conformal
p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,default=ROOT/'data'/'synthetic'); p.add_argument('--output',type=Path,default=ROOT/'outputs'/'benchmark'/'latency.json'); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
if not (a.data_dir/'continuous_logs.csv').exists():
    if a.data_dir.resolve() != (ROOT/'data'/'synthetic').resolve(): raise FileNotFoundError(a.data_dir)
    write_synthetic_inputs(a.data_dir,20260827)
opt=RuntimeOptions(erf_n_estimators=1,erf_max_depth=4,xgb_n_estimators=3,max_lith_rows_per_class=5,max_epochs=1) if a.smoke else RuntimeOptions()
ctx,folds,final=run_main_experiment(a.data_dir,options=opt,include_final_model=True); assert final is not None
cp=run_main_conformal(folds,final,alpha=0.05,seed=20260827); assets=prepare_latency_assets(ctx,final,q_hat=cp.blind_result.q_hat,options=opt); result=benchmark_end_to_end(assets,warmup_iterations=2 if a.smoke else 100,measured_iterations=5 if a.smoke else 1000)
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result.to_dict(),indent=2),encoding='utf-8'); print(a.output)
