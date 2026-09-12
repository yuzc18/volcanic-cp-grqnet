#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.analysis.conformal_analysis import VerificationCurve
from src.analysis.figures import figure12_verification_budget
p=argparse.ArgumentParser(); p.add_argument('--curve',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('outputs/figures/Fig12_verification_budget.png')); a=p.parse_args()
df=pd.read_csv(a.curve); kw={c:df[c].to_numpy() for c in df.columns if c!='budget_fraction'}; curve=VerificationCurve(budget_fraction=df['budget_fraction'].to_numpy(),**kw); figure12_verification_budget(curve,output=a.output); print(a.output)
