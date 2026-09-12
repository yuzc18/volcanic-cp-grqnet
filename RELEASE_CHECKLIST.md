# Public repository release checklist

Before publishing/updating GitHub:

1. Tag the release commit `v1.0`, record its SHA, and confirm both are reachable; cite that SHA in the finalized manuscript. Repository documentation refers to the tag, so no self-referential SHA has to be committed.
2. Run `pytest -q`.
3. Run `bash scripts/run_virtual_demo.sh` in a clean checkout/archive.
4. Run `python scripts/release_audit.py` and resolve every error.
5. Confirm no proprietary measurements, fitted study checkpoints, generated study outputs, or user-specific absolute paths are tracked.
6. Confirm `configs/search_spaces.yaml`, `metadata/per_fold_configs/`, `metadata/fold_assignments.csv`, and `metadata/split_schema.csv` are present; confirm that no reconstructed row-level `metadata/splits.csv` is shipped.
7. Prefer Python 3.10/3.11 with pinned `requirements.txt` for final CI.
8. If scientific source/configuration changes after the pinned commit, create and cite a new scientific revision.
