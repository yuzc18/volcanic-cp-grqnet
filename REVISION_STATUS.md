# Implementation status

The finalized manuscript is the specification for this repository.

**Scientific implementation revision:** release tag `v1.1` of this repository.

`v1.1` adds the fixes made in response to the 2026-09-13 repository review (`CODE_REVIEW_RESPONSE.md`). The audited predecessor `v1.0` stays reachable but does not contain them.

## Implemented manuscript workflow

- FZI calculation, K-means label construction, boundary bootstrap, single-start stability analysis, and alternative label definitions;
- 50-dimensional UFT with Groups A/B/C/D and the strictly causal Group D variant;
- outer leave-one-development-well-out evaluation and inner upstream leave-one-well-out cross-fitting;
- GRQ-Net with the manuscript Eq. (7)--(9) gated residual block;
- four neural and five non-neural baselines, complete HPO search spaces, Table 5 ablations, and the seven-setting capacity scan;
- randomized non-empty APS with fold-specific and deployment thresholds;
- Table 7 calibration-design refits;
- paired well-level intervals, moving-block bootstrap, Clopper-Pearson intervals, thinning, ACE, SHAP, Figure 6--12 generation, and six-stage CPU benchmarking;
- anonymized split/configuration metadata and synthetic end-to-end execution.

## Data boundary

No proprietary measurement values, fitted study checkpoints, or study prediction tables are published.

Public split metadata report the fold sizes and well-by-class calibration quotas in Table 3(b). A schema is provided for optional study-specific row mappings; the historical sample-level split assignment is not included in this release. There is no row-level table of the 178 calibration sample identities in this repository, and none is reconstructed.

## Runnable demonstration

```bash
bash scripts/run_virtual_demo.sh
```

The demonstration uses synthetic measurements and reduced smoke-test estimator sizes/epochs for speed. Its numerical results are intentionally not expected to match the manuscript.
