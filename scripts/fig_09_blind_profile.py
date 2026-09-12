#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.analysis.figures import figure9_blind_profile
p=argparse.ArgumentParser(); p.add_argument('--blind',type=Path,required=True); p.add_argument('--qhat',type=float,required=True); p.add_argument('--well',default='CS9'); p.add_argument('--output',type=Path,default=Path('outputs/figures/Fig09_CS9.png')); a=p.parse_args()
figure9_blind_profile(pd.read_csv(a.blind),q_hat=a.qhat,well_id=a.well,output=a.output); print(a.output)
