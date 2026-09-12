#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.analysis.figures import figure6_petrophysics
from src.data.labels import FZIBoundaries
p=argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, default=Path('outputs/figures/Fig06_petrophysics.png')); p.add_argument('--medium-low', type=float, required=True); p.add_argument('--high-medium', type=float, required=True); a=p.parse_args()
figure6_petrophysics(pd.read_csv(a.input), FZIBoundaries(a.medium_low,a.high_medium), output=a.output)
print(a.output)
