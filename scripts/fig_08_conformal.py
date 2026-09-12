#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.analysis.conformal_analysis import CoverageCurve
from src.analysis.figures import figure8_conformal_behavior
from src.analysis.statistics import average_coverage_error
p=argparse.ArgumentParser(); p.add_argument('--curve',type=Path,required=True); p.add_argument('--oof',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('outputs/figures/Fig08_conformal.png')); a=p.parse_args()
c=pd.read_csv(a.curve); curve=CoverageCurve(c['alpha'].to_numpy(),c['nominal_coverage'].to_numpy(),c['empirical_coverage'].to_numpy(),average_coverage_error(c['nominal_coverage'],c['empirical_coverage']))
figure8_conformal_behavior(curve,pd.read_csv(a.oof),output=a.output); print(a.output)
