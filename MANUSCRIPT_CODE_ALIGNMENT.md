# Manuscript-code alignment

The finalized manuscript is the authoritative specification for this repository.

**Scientific implementation revision:** release tag `v1.1` of this repository.

`v1.1` supersedes `v1.0` (the version audited by the 2026-09-13 review). `v1.0` remains reachable for traceability but does not contain the review fixes, so it must not be cited as the revision containing them.

This revision aligns the runnable implementation with the manuscript in the following places that are especially easy to audit:

- GRQ-Net gated residual Eq. (7) uses `ReLU(W_h x + b_h)` before Sigmoid gating; Eq. (9) applies dropout to the gated branch before residual addition and LayerNorm.
- Figure 8(a) displays nine levels sampled from the 30-point alpha grid and adds the `(1.00, 1.00)` trivial limit as a display-only reference; ACE is still computed over the 30-point grid only.
- Group B uses the five raw LWD curves, training-only z-score + Borderline-SMOTE, a 300-tree C4.5-style gain-ratio ensemble and Kendall-W stability guidance. The published reference-[50] candidate spaces are recorded in `configs/erf.yaml`; low-level choices not uniquely fixed by the paper are pinned in that file.
- `configs/search_spaces.yaml` is the authoritative executable source for the manuscript non-neural HPO protocol: 50 Optuna trials for LightGBM/XGBoost/CatBoost, four logistic-regression grid configurations, 18 random-forest grid configurations, internal 15% class-stratified selection, and macro-F1 as the selection metric. No undocumented alternative search space is used.
- `metadata/per_fold_configs/` defines the export contract for fold-specific HPO records; `scripts/run_baselines.py` writes the actually selected configuration for every outer fold and the separately tuned final 712-sample model. No Table 3(a) final configuration is copied into folds 1--5.
- `metadata/fold_assignments.csv` provides the anonymized, measurement-free fold/calibration metadata actually supported by Table 3(b): fold sizes and exact well-by-class calibration quotas. No reconstructed row-level manifest is distributed or represented as the historical fixed 178 sample identities. `metadata/split_schema.csv` documents the optional authoritative-ID input accepted by the code when such a record exists.
- Table 5 GRQ-Net ablations remove the omitted semantic group and its encoder and re-instantiate the model using only the active groups.

## Changes from the 2026-09-13 repository review

- Table 8 selects LightGBM hyperparameters per label definition and outer fold via `tune_non_neural`, with the selected configurations exported; the fixed-configuration branch is labelled as such and never as HPO. The class count follows the label definition, so the c=2 and c=4 rows are no longer fitted as three-class problems.
- The Table 8 ARI is computed on the 890 development samples used for label construction, and the population size is exported alongside it.
- Neural baselines are seeded immediately before model construction, so initial weights depend only on the requested seed.
- Every conformal analysis unit draws from its own named substream, so `scripts/eval_blind.py` and the deployment step of `run_main_conformal` return the same threshold and the same prediction sets.
- SHAP additivity is measured per fold and enforced by default; the residuals are published next to the attributions and reported separately from run completion.
- `RuntimeOptions.group_d_variant` propagates the centered/causal Group D choice through features, preprocessing and training, with a driver for the Section 4.3 comparison.
- `finite_sample_order_statistic` reports rather than clips the `n+1` case, with an explicit conservative full-label-set branch.

No proprietary well-log, core, thin-section, fitted-checkpoint, or measurement-bearing intermediate data are included.
