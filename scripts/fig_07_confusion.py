#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.analysis.figures import figure7_confusion_matrices
p=argparse.ArgumentParser(); p.add_argument('--baseline-predictions', type=Path, required=True); p.add_argument('--grq-oof', type=Path, required=True); p.add_argument('--output', type=Path, default=Path('outputs/figures/Fig07_confusion.png')); a=p.parse_args()
b=pd.read_csv(a.baseline_predictions); g=pd.read_csv(a.grq_oof); g=pd.DataFrame({'model':'grqnet','quality_class_id':g['quality_class_id'],'predicted_class_id':g['predicted_class_id']}); frame=pd.concat([b[['model','quality_class_id','predicted_class_id']],g],ignore_index=True)
figure7_confusion_matrices(frame, output=a.output); print(a.output)
