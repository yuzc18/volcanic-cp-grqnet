# Group-B eRF implementation note

Group B follows reference [50], as required by the finalized manuscript:

- five raw LWD inputs;
- z-score normalization;
- training-only Borderline-SMOTE;
- 300-tree enhanced random forest;
- C4.5-style gain-ratio splitting;
- Kendall's-W feature-stability guidance;
- 18-class probability output in the fixed class order in `configs/erf.yaml`.

Reference [50] additionally reports the following inner-CV candidate spaces: `n_estimators={100,200,300}`, `max_features={sqrt,log2}`, `n_bins={8,16,32}`, Borderline-SMOTE `k_neighbors={3,5,7}`, `m_neighbors={8,10,12}`, `kind={borderline-1,borderline-2}`, and stability-subset `MIN_TOPK={6,10}`, `MAX_TOPK={16,20}`. These are recorded in `configs/erf.yaml`.

The finalized paper fixes the deployed high-level eRF specification but does not uniquely identify every selected low-level value. The release therefore pins deterministic manuscript-consistent choices in `configs/erf.yaml` (`max_features=sqrt`, `n_bins=16`, `k_neighbors=5`, `m_neighbors=10`, `borderline-1`) so the complete pipeline is runnable and auditable. With five input curves, the published `sqrt` and `log2` choices both expose two candidate features per node.

The complete eRF pipeline is refitted under every active outer/inner/calibration design using the manuscript exclusion rules.
