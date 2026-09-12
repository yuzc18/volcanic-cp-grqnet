# Manuscript-code alignment

The finalized manuscript is the authoritative specification for this repository.

**Scientific implementation revision:** release tag `v1.0` of this repository.

This revision aligns the runnable implementation with the manuscript in the following places that are especially easy to audit:

- GRQ-Net gated residual Eq. (7) uses `ReLU(W_h x + b_h)` before Sigmoid gating; Eq. (9) applies dropout to the gated branch before residual addition and LayerNorm.
- Figure 8(a) displays nine levels sampled from the 30-point alpha grid and adds the `(1.00, 1.00)` trivial limit as a display-only reference; ACE is still computed over the 30-point grid only.
- Group B uses the five raw LWD curves, training-only z-score + Borderline-SMOTE, a 300-tree C4.5-style gain-ratio ensemble and Kendall-W stability guidance. The published reference-[50] candidate spaces are recorded in `configs/erf.yaml`; low-level choices not uniquely fixed by the paper are pinned in that file.
- `configs/search_spaces.yaml` is the authoritative executable source for the manuscript non-neural HPO protocol: 50 Optuna trials for LightGBM/XGBoost/CatBoost, four logistic-regression grid configurations, 18 random-forest grid configurations, internal 15% class-stratified selection, and macro-F1 as the selection metric. No undocumented alternative search space is used.
- `metadata/per_fold_configs/` defines the export contract for fold-specific HPO records; `scripts/run_baselines.py` writes the actually selected configuration for every outer fold and the separately tuned final 712-sample model. No Table 3(a) final configuration is copied into folds 1--5.
- `metadata/fold_assignments.csv` provides the anonymized, measurement-free fold/calibration metadata actually supported by Table 3(b): fold sizes and exact well-by-class calibration quotas. No reconstructed row-level manifest is distributed or represented as the historical fixed 178 sample identities. `metadata/split_schema.csv` documents the optional authoritative-ID input accepted by the code when such a record exists.
- Table 5 GRQ-Net ablations remove the omitted semantic group and its encoder and re-instantiate the model using only the active groups.

No proprietary well-log, core, thin-section, fitted-checkpoint, or measurement-bearing intermediate data are included.
