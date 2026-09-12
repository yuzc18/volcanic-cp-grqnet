# Code availability

**Name of code:** GRQ-Net + Conformal Prediction framework for volcanic reservoir quality evaluation (UFT + GRQ-Net + CP)

**Year first available:** 2026

**Hardware required:** Standard PC; no GPU required (~320K-parameter model; 0.49 ms classifier-only CPU inference, complete-pipeline median 2.846 ms, Table 6(b) of the manuscript).

**Software required:** Python 3.10–3.11; key dependencies include PyTorch 2.3.1, scikit-learn 1.4.2, XGBoost 2.0.3, LightGBM 4.3.0, CatBoost 1.2.5, pytorch-tabnet 4.1.0, Optuna, SHAP, NumPy, pandas, and SciPy; the complete pinned list is provided in `requirements.txt`.

**Program language:** Python

**Program size:** approximately 465 KB as the uncompressed source tree (119 tracked files) and approximately 180 KB as the compressed release archive; exact values vary slightly with packaging metadata. `python scripts/release_audit.py` reports the tracked size of the current checkout.

**Availability:** Public MIT-licensed code is available at `https://github.com/yuzc18/volcanic-cp-grqnet`, release tag `v1.0`. The repository includes training/evaluation code, the machine-readable 50-feature dictionary, anonymized fold/calibration metadata, random-seed settings, executable per-fold hyperparameter-configuration export, full machine-readable search spaces, statistical and figure-generation code, a synthetic-data generator, and the latency benchmark. Proprietary well-log, core, and thin-section measurements are not included.

**License:** MIT License
