# Response to the 2026-09-13 repository review

The review audited commit `ba580ed53b3c2264152fc0f80a95ad953b0f6754` (tag `v1.0`)
against the revised manuscript and raised eight items requiring change (C01-C08)
and one optional improvement (C09). This file records what changed in the code
and, separately, what still requires evidence that only the original study
materials can supply.

The distinction matters and is kept explicit throughout: **a fix to this
repository is not evidence about the historical run.** Where the review noted
that the effect on the paper depends on what the original code did, that
question is listed as open rather than answered.

## Summary

| Item | Subject | Code status | Still required from the authors |
|---|---|---|---|
| C01 | Table 8 GBM used a fixed configuration but was labelled HPO | Fixed | Confirm whether the historical Table 8 ran real per-fold selection |
| C02 | ARI computed on 1259 rows instead of 890 | Fixed | Recompute the reported ARI values from the existing labels |
| C03 | Neural baselines seeded after model construction | Fixed | Check whether the original run seeded at an outer level |
| C04 | Two blind-well CP entry points used different RNG positions | Fixed | Confirm which entry point produced the reported blind-well CP numbers |
| C05 | SHAP additivity check was disabled; attributions fail it | Diagnosed and enforced | Re-verify Figure 10 on the original checkpoint, or regenerate it |
| C06 | Causal Group D had no training entry point | Fixed | Compare against the historical 0.850 comparison if its outputs exist |
| C07 | Documentation claimed a sample-level manifest was published | Fixed | None |
| C08 | eRF provenance unclear | Documented | Locate the Group B implementation actually used for the manuscript |
| C09 | `n+1` order statistic silently clipped for small n | Fixed | None; no manuscript design reaches this branch |

## What changed

### C01 -- Table 8 selects LightGBM hyperparameters per fold

`scripts/run_label_sensitivity.py::_gbm_cv` called `fixed_reported_params` and
fitted directly, so no selection occurred on any path, while the output was
labelled `restored manuscript HPO`. The default path now calls
`tune_non_neural` for every (label definition, outer fold) pair, selecting on
that fold's internal 15% split under that definition's labels and refitting on
the outer-training subset -- the protocol `run_baselines.py` already used for
Table 6(a). `--reported-fixed-demo` is a separate branch reporting
`reported_table3a_fixed_demo_not_hpo`; the string `restored manuscript HPO` no
longer exists in the repository.

`tune_non_neural` now takes `n_classes` and passes it to every estimator it
builds, so the c=2 and c=4 definitions are no longer fitted as three-class
problems.

Each run writes `table8b_gbm_per_fold_configurations.csv`: one row per
(definition, fold) with the selection engine, the internal macro-F1 and the
selected parameters. A smoke run produces 20 such records.

### C02 -- ARI population

`adjusted_rand_score(adopted, labels_all)` became
`adjusted_rand_score(adopted[devmask], labels_all[devmask])`, and each row now
carries `n_ari_samples`, which is 890 on the manuscript split. The mask was
already computed in the same script.

On synthetic data the corrected values reproduce the review's 890-row column
exactly (K-means c=2: 0.312521; c=4: 0.353088; K-medoids: 0.819802).

### C03 -- Seeding before weight initialization

`nn.Module` draws its weights at construction, so a seed set inside a training
function cannot change them. `src/config.py::seed_before_model_construction` is
now called immediately before `build_neural_baseline` in
`src/training/baseline_runner.py` and `scripts/run_neural_seed_check.py`.
GRQ-Net's `train_one_fold` already seeded before construction and is unchanged.

Reproducing the review's probe: with ambient seeds 111 and 222 and the same
requested seed 20260827, the initial weight hashes are now identical, while
different requested seeds still give different weights.

### C04 -- One deployment CP result

`run_main_conformal` drew from a single sequential `default_rng(seed)` stream
across five folds and then the blind wells, while `scripts/eval_blind.py`
started a fresh stream at the blind step. Identical probabilities therefore gave
q-hat 0.9624932 from one entry point and 0.9776393 from the other, with 93 of
369 prediction sets differing.

Each analysis unit now draws from its own named substream,
`conformal_substream(seed, unit)`, derived from the seed and a SHA-256 digest of
the unit name (`outer_fold_1` ... `outer_fold_5`, `deployment_blind`). Both
entry points request `DEPLOYMENT_UNIT`, so they produce the same q-hat and the
same sets, and a unit's result no longer depends on how many other units ran
first.

This changes which uniforms the deployment step consumes relative to the
previous sequential stream. It is a change of stream organisation, not of the
APS rule: the score definition, the ordered threshold and the non-empty
guarantee are untouched. Reported blind-well CP quantities should be recomputed
from the saved final-model probabilities; no classifier needs retraining.

### C05 -- SHAP additivity

`explainer.shap_values(..., check_additivity=False)` suppressed the diagnostic,
so the script completed and wrote Figure 10 regardless of validity.
`compute_foldwise_shap` now computes the residual itself -- SHAP values plus the
explainer's expected value against the model logits -- records it per fold, and
by default raises `ShapAdditivityError` rather than returning attributions that
fail. `run_round6_analysis.py` always writes
`fig10_shap_additivity_diagnostic.csv` and reports
`shap_additivity_verified` in its manifest, separately from
`shap_completed`.

The cause was isolated on minimal control models with the pinned stack
(SHAP 0.45.1, PyTorch 2.3.1):

| Control model | max residual | verdict |
|---|---:|---|
| `Linear` | 3.0e-07 | pass |
| `Linear -> nn.ReLU -> Linear` | 1.1e-07 | pass |
| `Linear -> nn.GELU -> Linear` | 2.9e-01 | fail |
| `Linear -> nn.LayerNorm -> Linear` | 2.1e-01 | fail |
| `Linear -> F.relu -> Linear` | 3.8e-01 | fail |

There are two independent causes, and both are present in GRQ-Net. SHAP 0.45.1
has no DeepLIFT rule for `nn.GELU` or `nn.LayerNorm`, which is the source of its
`unrecognized nn.Module` warnings; and DeepExplainer hooks `nn.Module`
instances, so the functional `F.relu` and `torch.sigmoid` calls in
`FullContextGroupAttention` and `GatedResidualBlock` are not attributed at all.
The last table row shows a network failing with `F.relu` where the identical
network passes with `nn.ReLU`. Exposing the activations as modules would address
the second cause only.

This is a property of the explainer and the architecture, not of any particular
training run, so it applies to any GRQ-Net explained this way with this stack.
Whether the published Figure 10 is affected depends on the explainer and
versions actually used, which this repository cannot determine.

Nothing was removed from the interpretability discussion: group weights and SHAP
remain distinct quantities, and the ablations remain the evidence for a group's
contribution.

### C06 -- Causal Group D reaches training

`RuntimeOptions` gains `group_d_variant` (`"centered"` default, or `"causal"`),
validated on construction and exposed as `group_d_causal`. Both training
preparation paths in `src/pipeline.py` now pass it to
`build_group_d_continuous` instead of the hard-coded `causal=False`, so the
variant propagates through Group D, the preprocessing fitted on it, and GRQ-Net
training.

`scripts/run_group_d_variant_comparison.py` runs both arms over the same folds
and seed and writes per-fold macro-F1, the paired per-fold difference and the
mean difference in percentage points. No source editing is required. The
centered default is unchanged.

### C07 -- Manifest description

`REVISION_STATUS.md` and the `src/uncertainty/calibration_designs.py` header
said the release ships an anonymized split manifest with sample tokens and role
assignments. It does not. Both now state that the published metadata are the
Table 3(b) fold sizes and well-by-class calibration quotas plus a schema for an
optional study-specific row mapping, and that the historical sample-level
assignment is not included. The README's `model-configuration records` phrase
now names what is actually shipped: configuration files, the complete search
spaces, and the runtime export interface.

### C08 -- eRF provenance

`UPSTREAM_ERF_NOTE.md` was rewritten to separate three layers: the published
algorithm of reference [50], this release's re-implementation, and the
implementation actually used for the manuscript. The release is now labelled a
public re-implementation written for release, and the pinned low-level values
are labelled release choices rather than recovered settings.

The note also describes what the stability mechanism does here:
`_consensus_weights` turns the mean gain-ratio rank over previously grown trees
into per-feature split weights, while Kendall's W is computed once after fitting
and stored as the `kendall_w_` diagnostic without feeding back. It states that a
different realisation would also be consistent with the published description,
and that the function name does not establish equivalence.

This is a documentation change. The correspondence itself remains open.

### C09 -- Small calibration sets

`finite_sample_order_statistic` clipped `ceil((n+1)(1-alpha))` to `n`, returning
an ordinary score while still carrying the finite-sample statement. It now
raises `InsufficientCalibrationError`, naming the required rank and the minimum
calibration size for that alpha. `compute_threshold` accepts
`on_insufficient_calibration="full_set"`, which returns `+inf`, and
`build_prediction_sets_detailed` accepts an infinite threshold by returning the
full label set.

The 30-point ACE sweep uses the conservative branch, because alpha=0.01 needs
n>=99 and the grid is also applied to small demonstration sets. No manuscript
design reaches the branch: the primary analysis uses n=178 over alpha=0.01-0.30
and the smallest Table 7 design is n=89 at alpha=0.05.

## Regression coverage

`tests/test_review_regressions.py` adds 22 tests covering exactly the checks the
review asked for: the Table 8 default path calls the tuner and the fixed branch
is not labelled HPO; the tuner honours c=2 and c=4; the ARI uses the development
mask and exports its population size; initial weights are independent of the
ambient random state for two baselines while still responding to the requested
seed; both CP entry points agree on q-hat and on every prediction set; the SHAP
additivity diagnostic exists and is enforced, with the control models
documenting the cause; the Group D variant is validated and reaches both
training preparation paths; and the small-n branch is reported rather than
clipped.

Full suite: 91 passed. The review's twelve smoke commands still exit 0, plus the
new comparison driver.

## Open items

These cannot be closed from this repository:

1. **C05** -- re-run the additivity check on the checkpoint, SHAP version and
   background actually used for Figure 10. If it fails there too, regenerate the
   affected attributions with a verified explainer.
2. **C08** -- locate the Group B implementation or configuration used for the
   manuscript and compare it against this release.
3. **C01, C02, C03, C04, C06** -- determine what the historical run did, and
   recompute only the affected quantities. By the review's own analysis none of
   these requires retraining the GRQ-Net main results: C02 is label
   post-processing, C04 is APS post-processing on saved probabilities, and C01
   affects the Table 8 GBM comparison only if the historical run also used a
   fixed configuration.

Until those are settled, statements such as "all reported analyses are
reproduced" or "all numerical results are independently verified" are not
supported. What this release supports is the implementation, the configuration,
the run steps, and their auditability.
