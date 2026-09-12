# UFT + GRQ-Net + Conformal Prediction for Volcanic-Reservoir Quality Evaluation

Public code corresponding to the finalized manuscript.

**Scientific implementation revision:** release tag `v1.0` of this repository. The release is a single commit; the manuscript cites that commit's SHA, which `v1.0` points to.

The manuscript is the specification. The code implements the workflow described there:
five standard logging-while-drilling (LWD) curves are transformed into a 50-dimensional
unified feature table (UFT), classified with GRQ-Net, and converted into randomized
adaptive prediction sets (APS). Proprietary study measurements are not distributed.

The revision above is the paper-facing scientific implementation. Later documentation-only
commits may improve release presentation without changing the scientific workflow.

## What is public

The repository provides:

- the complete training/evaluation pipeline;
- the machine-readable 50-feature dictionary;
- documented split-generation and calibration-design procedures;
- random-seed settings and model/configuration files;
- statistical analyses and Figure 6--12 generation code;
- a synthetic-data generator for end-to-end execution without proprietary data;
- the six-stage CPU latency benchmark.

The repository does **not** contain proprietary well-log, core, or thin-section measurements,
measurement-bearing intermediate files, fitted study checkpoints, or study prediction tables.
It does include anonymized, measurement-free fold/calibration metadata at the level reported in
Table 3(b) (fold sizes and well-by-class calibration quotas), plus model-configuration
records required by the finalized manuscript.

Synthetic outputs are demonstration values. They are not expected to reproduce the
numerical results reported in the manuscript.

## Quick start: runnable synthetic demonstration

Create a Python 3.10--3.11 environment and install the pinned dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

Then run the end-to-end synthetic demonstration:

```bash
bash scripts/run_virtual_demo.sh
```

The demo generates synthetic continuous LWD curves, core measurements, and an 18-class
lithology corpus, then executes the manuscript data flow:

```text
five LWD curves
    -> FZI labels
    -> Group A (5)
    -> Group B eRF probabilities (18)
    -> Group C XGBoost proxies (2)
    -> Group D local context (25)
    -> 50-D UFT
    -> 178 calibration / 712 downstream modeling split
    -> outer leave-one-well-out + inner upstream cross-fitting
    -> GRQ-Net
    -> randomized non-empty APS
    -> blind-well evaluation
```

For speed, smoke/demo commands reduce estimator sizes and neural epochs only for the
execution check. The manuscript-facing settings remain in the YAML configuration files.

Run the unit/integration tests with:

```bash
pytest -q
```

## Data interface

The public interface separates three inputs:

```text
<input-directory>/
  continuous_logs.csv   # continuous 0.125 m LWD grid
  core_samples.csv      # core-matched porosity/permeability rows
  lithology_corpus.csv  # lithology-labelled rows for Group B
```

`scripts/generate_synthetic_data.py` writes this three-file set to `data/synthetic/`,
which is the input directory used by the demonstration commands. User-supplied data may
live in any directory with the same schema.

Full column definitions and units are documented in `data/README.md`.

The five LWD inputs are:

```text
GR, CNL, DEN, AC, RLA5
```

No additional logging channel is required at inference.

## Manuscript-to-code correspondence

### Reservoir-quality labels

FZI is computed from the recorded core porosity and permeability. K-means is fitted in
`log10(FZI)` space on the 890 development-well samples with k-means++ initialization,
`n_init=10`, `max_iter=300`, `tol=1e-4`, and `random_state=42`. The adopted boundaries
are frozen for downstream use. Boundary bootstrap, the 50-run single-start stability
analysis, the `c=2...6` validity scan, and the K-means/K-medoids alternatives are implemented
in the label/sensitivity modules and `scripts/run_label_sensitivity.py`.

### 50-dimensional UFT

The UFT group dimensions are fixed at `5 / 18 / 2 / 25`:

```text
A: GR, CNL, DEN, AC, RLA5
B: LithProb_0 ... LithProb_17
C: predicted porosity, predicted log10-permeability
D: prev1, next1, mean3, std3(ddof=1), diff1 x five curves
```

Group D is computed on the continuous 0.125 m log sequence after 3-sigma truncation and
before matching to core depths. The strictly causal variant replaces the forward-looking
slots with the trailing-window definitions stated in the manuscript. Groups A, C, and D
are robust-scaled with training-fitted median/IQR parameters; Group B remains a probability
vector.

The complete 50-column definition is in `metadata/feature_dictionary.csv`.

### Group B: eRF lithology probabilities

Group B follows the published eRF specification cited in the manuscript: 300 trees,
training-only Borderline-SMOTE, C4.5 gain-ratio splitting, and Kendall's-W feature-stability
guidance. The complete fitting pipeline includes z-score normalization and is refitted for
each active split. The 18-class order is fixed in `configs/erf.yaml`.

Implementation details not uniquely fixed by the cited eRF publication are pinned and
documented in `configs/erf.yaml` and `UPSTREAM_ERF_NOTE.md` so the published specification is runnable.

### Group C: petrophysical proxies

Two XGBoost regressors predict recorded porosity and `log10(permeability)` from the same
five original LWD curves. The manuscript-facing fixed configuration is:

```text
300 boosting rounds
max_depth = 4
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

### Leakage-safe upstream fitting and downstream splits

The main split contains 890 development-well samples, of which 178 form the fixed CP
calibration subset and 712 form the downstream modeling pool. The two blind wells contain
216 and 153 samples.

The repository publishes the exact Table 3(b) calibration quotas but does not publish or reconstruct
a row-level list claiming to be the historical 178 proprietary sample identities. With user-supplied
data, the default implementation applies the published within-well/class stratified protocol using the
primary seed; an authoritative anonymized 178-ID file can instead be supplied through
`--split-metadata` when such a record is available.

The outer validation sizes are:

```text
WF1   128
CS12  114
CS602 206
CS607 125
CS11  139
```

For the downstream classifier's own training rows, Groups B and C are generated by inner
leave-one-well-out cross-fitting. Each outer validation well is excluded from the upstream
models that generate its proxies. The final blind-well upstream models are fitted on the
712 non-calibration development rows.

Neural early stopping uses a class-stratified 15% internal split, giving the manuscript
counts `88/90/76/89/86` for the five outer folds and `107` rows for the final 712-sample
model. The best neural checkpoint is retained without full-subset refitting.

### GRQ-Net

The paper-facing GRQ-Net configuration is:

```text
group embedding d_g = 48
reweighting hidden d_r = 32
four gated residual blocks
fused dimension = 192
dropout = 0.12
classification head = 192 -> 64 -> 3
trainable parameters = 319,815
```

Group weights are independent Sigmoid outputs and are exported as an auxiliary output.

### Baselines and Table 5 ablations

The four neural baselines and five non-neural baselines described in the manuscript are
implemented. The reported final configurations are in `configs/baselines.yaml`.

The executable tuning framework implements the manuscript selection protocol: 50 Optuna
trials for LightGBM/XGBoost/CatBoost and internal 15% macro-F1 selection, plus four
logistic-regression and 18 random-forest grid configurations. The authoritative executable
search spaces are in `configs/search_spaces.yaml`; this is the only search-space source read
by the manuscript-HPO path. Fresh runs export the actual selected configuration for each fold.
The release does not use an undocumented fallback or hidden alternative search domain.
`--reported-fixed-demo` remains available only as a fast fixed-configuration smoke path.

For Table 5 GRQ-Net ablations, an omitted semantic group and its encoder are removed and
the network is re-instantiated using only the active groups. All remaining architecture and
training settings are unchanged.

### Randomized APS conformal prediction

The implementation uses the manuscript randomized APS score, exact finite-sample ordered
threshold, randomized boundary removal, and a top-1 non-empty guard. For the primary
`n=178`, `alpha=0.05` design, the threshold is the 171st ordered calibration score.
Fold-specific thresholds are used for the five outer validation wells and a separate
deployment threshold is used for the blind wells. The main random seed is `20260827`.

Table 7 calibration designs refit the upstream and downstream pipeline for each alternative
calibration set while keeping only the FZI boundaries fixed.

### Statistics, interpretability, figures, and timing

The repository implements:

- paired per-well t-based intervals (`n=5`);
- well-stratified paired 2.0 m moving-block bootstrap intervals;
- Clopper-Pearson coverage reference intervals;
- 0.5 m / 1.0 m two-start greedy thinning without retraining;
- the 30-point `alpha=0.01...0.30` ACE analysis;
- Figure 12 random ranking with `R=200` and `B=1000` paired block-bootstrap replicates;
- fold-wise DeepExplainer SHAP on model logits with 100 training-background samples;
- the six-stage batch-one CPU benchmark using the strictly causal Group D variant.

All figure annotations are computed from supplied predictions. Manuscript result values are
not inserted into plotting code to force agreement.

## Paper-to-code map

| Manuscript item | Main code/configuration | Typical generated output |
|---|---|---|
| Table 1 | `configs/wells.yaml`; `scripts/export_core_summary_tables.py` | `table1_well_overview.csv` |
| Table 2(a) | `metadata/feature_dictionary.csv`; `src/features/uft.py` | 50-column UFT |
| Table 2(b) | `src/features/group_d.py`; `configs/features.yaml` | 25-D Group D |
| Table 3(a) | `configs/grqnet.yaml`, `configs/baselines.yaml`, `configs/upstream_xgb.yaml` | run metadata/config records |
| Table 3(b) | `src/data/splits.py`; `configs/data.yaml` | split roles/count checks |
| Table 4 | `src/analysis/tables.py`; `scripts/export_core_summary_tables.py` | `table4_petrophysical_summary.csv` |
| Table 5 | `src/analysis/ablation.py`; `scripts/run_ablation.py` | ablation JSON/CSV |
| Table 6(a) | `src/models/baselines.py`; `scripts/run_baselines.py` | OOF predictions/metrics |
| Table 6(b) | `src/analysis/benchmark.py`; `benchmark/run_latency.py` | latency JSON |
| Table 7 | `src/uncertainty/calibration_designs.py`; `scripts/run_table7_calibration_designs.py` | calibration-design summaries |
| Table 8(a,b) | `src/analysis/label_sensitivity.py`; `scripts/run_label_sensitivity.py` | Table 8 CSVs |
| Table 9(a,b) | `scripts/eval_blind.py`; `src/analysis/tables.py` | classification/coverage CSVs |
| Fig. 6 | `scripts/fig_06_petrophysics.py` | petrophysical figure |
| Fig. 7 | `scripts/fig_07_confusion.py` | confusion matrices |
| Fig. 8 | `scripts/fig_08_conformal.py` | calibration/set-size figure |
| Fig. 9 | `scripts/fig_09_blind_profile.py` | CS9 profile |
| Fig. 10 | `scripts/fig_10_interpretability.py` | SHAP/group-weight figure |
| Fig. 11 | `scripts/fig_11_fzi_sets.py` | FZI/set-size figure |
| Fig. 12 | `scripts/fig_12_verification_budget.py` | verification-budget curves |

## Reference environment and latency

The manuscript reference software environment is Python 3.10--3.11 with the direct
versions pinned in `requirements.txt`. The reported latency values were measured on a
single Intel Core i7-13700H core under Windows 11 with Python 3.11.9, PyTorch 2.3.1 CPU,
batch size one, and single-thread library settings. Timing on other hardware/software is a
valid code check but should not be compared numerically with the manuscript timing table.

## Code availability

Paper-facing release coordinates:

```text
Repository: https://github.com/yuzc18/volcanic-cp-grqnet
Scientific revision: tag v1.0
License: MIT
```

See `CODE_AVAILABILITY.md` for the manuscript-facing release description.
