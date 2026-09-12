#!/usr/bin/env python3
"""Audit the public code release without reading proprietary study data.

The audit checks the Git-tracked/release tree for required reproducibility
artifacts and prevents raw study data, generated outputs, checkpoints and user
absolute paths from entering the public repository. Identity is not treated as a
release error because this repository is not anonymized.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md", "LICENSE", "requirements.txt", "CODE_AVAILABILITY.md",
    "MANUSCRIPT_CODE_ALIGNMENT.md", "UPSTREAM_ERF_NOTE.md",
    "metadata/feature_dictionary.csv", "metadata/split_schema.csv",
    "metadata/fold_assignments.csv",
    "metadata/per_fold_configs/README.md",
    "metadata/per_fold_configs/schema.json",
    "metadata/per_fold_configs/manifest.yaml",
    "metadata/calibration_designs/schema.csv", "configs/seeds.yaml",
    "configs/search_spaces.yaml", "configs/conformal.yaml", "configs/erf.yaml",
    "benchmark/run_latency.py",
]

FORBIDDEN_TRACKED_PREFIXES = (
    "data/raw/", "outputs/", "checkpoints/", "results/", "figures/", "tables/",
)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".csv", ".json", ".sh"}
WIN_USER_PATH_RE = re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.I)
POSIX_USER_PATH_RE = re.compile(r"/(?:home|Users)/[^/\s]+")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def in_git_checkout() -> bool:
    return (ROOT / ".git").exists()


def tracked_files() -> list[str]:
    if in_git_checkout():
        return [x for x in git("ls-files").splitlines() if x]
    skip_parts = {"__pycache__", ".pytest_cache", ".venv", "venv"}
    out = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(part in skip_parts for part in p.relative_to(ROOT).parts):
            continue
        out.append(p.relative_to(ROOT).as_posix())
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None, help="Optional JSON report path")
    args = ap.parse_args()

    tracked = tracked_files()
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED:
        if rel not in tracked:
            errors.append(f"required release artifact missing: {rel}")

    for rel in tracked:
        if rel.startswith(FORBIDDEN_TRACKED_PREFIXES) and not rel.endswith(".gitkeep"):
            errors.append(f"forbidden tracked data/output artifact: {rel}")
        if rel.startswith("data/synthetic/") and not rel.endswith(".gitkeep"):
            errors.append(f"generated synthetic CSVs must be regenerated, not committed: {rel}")

    for rel in tracked:
        p = ROOT / rel
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if WIN_USER_PATH_RE.search(text) or POSIX_USER_PATH_RE.search(text):
            errors.append(f"absolute user path in {rel}")

    search_cfg_path = ROOT / "configs" / "search_spaces.yaml"
    search_cfg = search_cfg_path.read_text(encoding="utf-8")
    if "space: null" in search_cfg or "available: false" in search_cfg:
        errors.append("one or more manuscript non-neural HPO search spaces are unavailable")
    else:
        import yaml
        hpo = yaml.safe_load(search_cfg)
        sp = hpo.get("search_protocol", {})
        for name in ("lightgbm", "xgboost", "catboost"):
            entry = sp.get(name, {})
            if entry.get("engine") != "optuna" or int(entry.get("trials_per_outer_fold", -1)) != 50:
                errors.append(f"{name} must use Optuna with 50 trials per outer fold")
        lr = sp.get("multinomial_logistic_regression", {})
        if lr.get("engine") != "grid" or int(lr.get("n_configurations", -1)) != 4 or len(lr.get("space", [])) != 4:
            errors.append("multinomial logistic regression must expose exactly four grid configurations")
        rf = sp.get("random_forest", {})
        if rf.get("engine") != "grid" or int(rf.get("n_configurations", -1)) != 18 or len(rf.get("space", [])) != 18:
            errors.append("random forest must expose exactly 18 grid configurations")
        if hpo.get("selection_metric") != "macro_f1":
            errors.append("non-neural HPO selection metric must be macro_f1")
        if hpo.get("selection_split") != "internal_15_percent_stratified":
            errors.append("non-neural HPO selection split must be internal_15_percent_stratified")
        if hpo.get("final_non_neural_protocol") != "tune_on_712_then_refit_on_all_712":
            errors.append("final non-neural protocol must tune on 712 then refit on all 712")
    forbidden_per_fold_release = {
        "metadata/per_fold_configs/per_fold_configurations.csv",
        "metadata/per_fold_configs/study_reference.yaml",
    }
    for rel in forbidden_per_fold_release:
        if rel in tracked:
            errors.append(f"pseudo study per-fold record must not be committed: {rel}")

    provenance_scan = [
        rel for rel in tracked
        if rel.startswith("metadata/per_fold_configs/") or rel in {"README.md", "MANUSCRIPT_CODE_ALIGNMENT.md", "CODE_AVAILABILITY.md"}
    ]
    bad_markers = ("Table_3a_reference_" + "reconstruction", "manuscript_consistent_" + "reconstruction_from_Table_3a")
    for rel in provenance_scan:
        p = ROOT / rel
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in bad_markers):
            errors.append(f"Table 3(a)-copied pseudo per-fold configuration marker in {rel}")

    manifest_path = ROOT / "metadata" / "per_fold_configs" / "manifest.yaml"
    if manifest_path.exists():
        import yaml
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_of_selected_values") != "actual_fold_specific_hpo_run":
            errors.append("per-fold configuration manifest must require actual fold-specific HPO output")
        folds = manifest.get("outer_folds", [])
        expected_wells = ["WF1", "CS12", "CS602", "CS607", "CS11"]
        if [x.get("validation_well") for x in folds] != expected_wells:
            errors.append("per-fold export manifest has the wrong outer-fold well order")
        if not manifest.get("final_model", {}).get("refit_on_all_712", False):
            errors.append("per-fold export manifest must retain the final tune-then-refit protocol")

    # Never ship a reconstructed row-level split manifest. The release metadata
    # must remain at the aggregate Table 3(b) level unless a genuinely
    # authoritative anonymized 178-ID record is available.
    if "metadata/splits.csv" in tracked:
        errors.append("metadata/splits.csv must not ship reconstructed/pseudo row-level calibration identities")

    import pandas as pd
    folds = pd.read_csv(ROOT / "metadata/fold_assignments.csv")
    if int(folds["calibration_n"].sum()) != 178:
        errors.append("fold_assignments calibration_n must sum to 178")
    if int(folds["calibration_H"].sum()) != 51 or int(folds["calibration_M"].sum()) != 49 or int(folds["calibration_L"].sum()) != 78:
        errors.append("fold_assignments calibration class totals must be 51/49/78")
    if folds["validation_n"].astype(int).tolist() != [128, 114, 206, 125, 139]:
        errors.append("fold_assignments validation sizes do not match Table 3(b)")
    if folds["outer_training_n"].astype(int).tolist() != [584, 598, 506, 587, 573]:
        errors.append("fold_assignments outer-training sizes do not match Table 3(b)")

    if in_git_checkout():
        revision = git("rev-parse", "HEAD")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
    else:
        revision = "not-embedded-in-archive"
        branch = "archive"
    tracked_bytes = sum((ROOT / rel).stat().st_size for rel in tracked if (ROOT / rel).is_file())
    report = {
        "status": "PASS" if not errors else "FAIL",
        "revision": revision,
        "branch": branch,
        "tracked_file_count": len(tracked),
        "tracked_tree_bytes": tracked_bytes,
        "tracked_tree_kib": round(tracked_bytes / 1024.0, 1),
        "errors": errors,
        "warnings": warnings,
        "python": os.sys.version.split()[0],
    }
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
