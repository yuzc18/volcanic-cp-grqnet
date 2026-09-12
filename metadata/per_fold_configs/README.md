# Per-fold non-neural HPO configuration export

Section 3.6 of the manuscript selects the five non-neural baselines separately
inside each outer training fold. The executable source of those selected
configurations is the HPO path in `scripts/run_baselines.py`; selected values are
never copied from Table 3(a) into outer-fold records.

Running

```bash
python scripts/run_baselines.py --models lightgbm multinomial_logistic_regression random_forest xgboost catboost --include-final-non-neural --output-dir outputs/baselines
```

writes `outputs/baselines/per_fold_configurations.csv`. Each row is produced by
an actual fold-specific selection run on that run's input data and contains the
selection engine, internal 15% macro-F1, selected hyperparameters, and seed. The
five `final` rows are produced by the separate 712-sample final-model tuning path
followed by refitting on all 712 samples.

This public repository does **not** pre-populate study-specific selected values
by copying the Table 3(a) final blind-well configuration into folds 1--5. Such a
copy would not be evidence of the fold-specific HPO stated in the manuscript.
`schema.json` and `manifest.yaml` define the machine-readable export contract.

The proprietary measurements used in the study are not distributed, so a fresh
public run on the supplied synthetic data will generate valid per-fold records
for the synthetic demonstration, not the numerical study records.
