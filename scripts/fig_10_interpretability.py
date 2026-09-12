#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.analysis.figures import figure10_interpretability
p=argparse.ArgumentParser(); p.add_argument('--feature-shap',type=Path,required=True); p.add_argument('--group-shap',type=Path,required=True); p.add_argument('--oof',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('outputs/figures/Fig10_interpretability.png')); a=p.parse_args()
figure10_interpretability(pd.read_csv(a.feature_shap),pd.read_csv(a.group_shap),pd.read_csv(a.oof),output=a.output); print(a.output)
