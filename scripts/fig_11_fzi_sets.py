#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.analysis.figures import figure11_fzi_prediction_sets
from src.data.labels import FZIBoundaries
p=argparse.ArgumentParser(); p.add_argument('--oof',type=Path,required=True); p.add_argument('--medium-low',type=float,required=True); p.add_argument('--high-medium',type=float,required=True); p.add_argument('--output',type=Path,default=Path('outputs/figures/Fig11_fzi_sets.png')); a=p.parse_args()
figure11_fzi_prediction_sets(pd.read_csv(a.oof),FZIBoundaries(a.medium_low,a.high_medium),output=a.output); print(a.output)
