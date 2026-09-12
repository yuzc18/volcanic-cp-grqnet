# Metadata

This directory contains non-proprietary metadata accompanying the finalized manuscript.

- `feature_dictionary.csv`: machine-readable definitions of all 50 UFT features.
- `fold_assignments.csv`: the five outer-fold sizes, internal early-stopping counts, and the exact per-well/per-class calibration quotas reported in Table 3(b). This file is the released anonymized fold/calibration metadata for the main analysis.
- `split_schema.csv`: schema for an **optional authoritative row-level split file**. Such a file is not included because the historical proprietary sample-ID mapping is not available in the release materials. If the authors/users possess an authoritative anonymized ID list, the training scripts can ingest it and validate the published 178-row total and Table 3(b) cell quotas.
- `calibration_designs/`: machine-readable calibration-design schema for Table 7.
- `per_fold_configs/`: machine-readable contract for the real fold-specific HPO records exported by `scripts/run_baselines.py`; no Table-3(a)-copied pseudo-fold records are committed.

No reconstructed row-level manifest is distributed. In particular, the repository does **not** create anonymous placeholder rows and present them as the historical fixed 178 calibration samples.
