# Public repository release checklist

Before publishing/updating GitHub:

1. Tag the release commit, record its SHA, and confirm both are reachable; cite that SHA in the finalized manuscript. Repository documentation refers to the tag, so no self-referential SHA has to be committed.
2. Run `pytest -q`.
3. Run `bash scripts/run_virtual_demo.sh` in a clean checkout/archive.
4. Run `python scripts/release_audit.py` and resolve every error.
5. Confirm `shap_additivity_verified` in `analysis_manifest.json`. Figure-10 attributions produced with `--shap-additivity record` are for inspection only and must not be presented as validated; `fig10_shap_additivity_diagnostic.csv` carries the per-fold residuals.
6. Run `python scripts/run_group_d_variant_comparison.py` if the Section 4.3 centered-versus-causal comparison is being refreshed.
7. Confirm no proprietary measurements, fitted study checkpoints, generated study outputs, or user-specific absolute paths are tracked.
8. Confirm `configs/search_spaces.yaml`, `metadata/per_fold_configs/`, `metadata/fold_assignments.csv`, and `metadata/split_schema.csv` are present; confirm that no reconstructed row-level `metadata/splits.csv` is shipped.
9. Prefer Python 3.10/3.11 with pinned `requirements.txt` for final CI.
10. If scientific source/configuration changes after the pinned commit, create and cite a new scientific revision.
11. Check `CODE_REVIEW_RESPONSE.md` for items still marked open, and do not describe those as closed.
