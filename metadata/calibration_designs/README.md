# Calibration-design metadata

`metadata/fold_assignments.csv` contains the published main-design fold sizes and per-well/per-class calibration quotas. `schema.csv` defines the row-level format used when Table-7 alternative designs are exported from an actual run.

The Round-5 driver writes the active design indices under the gitignored `outputs/table7/` directory so every supplied or synthetic run can be audited.

Implemented Table-7 families:

- within-well class-stratified 10/15/20/25% designs (n=89/134/178/223; 10 fixed seeds);
- 20% contiguous depth-ordered blocks (n=178; 5 fixed seeds);
- whole-well calibration on CS12, CS607, and WF1 with the manuscript evaluation targets.

The released fixed repeat-seed policy is `20260827 + repeat_index`; the explicit ten-seed and five-seed lists are recorded in `configs/conformal.yaml`.
