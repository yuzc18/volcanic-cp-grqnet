# Group-B eRF implementation note

## Status of this implementation

**This is a public re-implementation written for release, not the original
study code.** It follows the specification the manuscript attributes to
reference [50], but it was not recovered from the environment that produced the
reported Group B probabilities. Nothing in this file should be read as a
reconstruction of the original run's low-level settings.

Readers auditing the paper should keep three things apart:

| Layer | What it is | Where it lives |
|---|---|---|
| Published algorithm | The high-level eRF method of reference [50]: five raw LWD inputs, z-score normalization, training-only Borderline-SMOTE, a 300-tree forest with C4.5 gain-ratio splitting, Kendall's-W feature-stability guidance, 18-class output | [Reference 50](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335630) |
| This release | A runnable re-implementation of that algorithm, with the values below pinned so the pipeline executes deterministically | `src/features/group_b.py`, `configs/erf.yaml` |
| The original study | The Group B implementation actually used to produce the manuscript's numbers | Not in this repository |

The first and second layers are documented here. The correspondence between the
second and the third has not been established, and this release does not claim
it.

## What reference [50] fixes, and what this release pins

Reference [50] reports these inner-CV candidate spaces: `n_estimators={100,200,300}`,
`max_features={sqrt,log2}`, `n_bins={8,16,32}`, Borderline-SMOTE `k_neighbors={3,5,7}`,
`m_neighbors={8,10,12}`, `kind={borderline-1,borderline-2}`, and stability-subset
`MIN_TOPK={6,10}`, `MAX_TOPK={16,20}`. These are recorded in `configs/erf.yaml`.

The paper fixes the deployed high-level specification but does not uniquely
identify every selected low-level value. This release therefore pins
deterministic, specification-consistent choices (`max_features=sqrt`,
`n_bins=16`, `k_neighbors=5`, `m_neighbors=10`, `borderline-1`) so the complete
pipeline is runnable and auditable. **These pinned values are choices made for
this release; they are not values recovered from the original study.** With five
input curves, the published `sqrt` and `log2` options both expose two candidate
features per node.

## How the stability mechanism works here

`KendallGuidedC45Forest` (`src/features/group_b.py`) grows trees sequentially.
Before each tree, `_consensus_weights` converts the mean gain-ratio rank of the
features across all previously grown trees into per-feature weights, so features
that earlier trees ranked consistently high are favoured at split selection. The
Kendall coefficient of concordance itself is computed once, after the whole
forest is grown, and is stored as the `kendall_w_` diagnostic; it does not feed
back into fitting.

This is a specific reading of "Kendall's-W stability guidance". A different
implementation -- for example one that recomputes W during fitting and uses it
to gate a feature subset -- would also be consistent with the published
description. **The function name alone does not establish equivalence with the
original implementation**, and the correspondence should be checked against the
original Group B code before any claim of equivalence is made.

## What would close this

Recovering the Group B implementation or configuration actually used for the
manuscript, and comparing its settings and stability mechanism against this
file. If they agree, record the source and parameters here. If they differ,
the affected Group B results are the ones to re-examine; this note should then
say plainly that the release is an independent re-implementation.

## Refitting scope

The complete eRF pipeline is refitted under every active outer/inner/calibration
design using the manuscript exclusion rules.
