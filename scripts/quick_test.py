"""Foundation smoke test for configuration, metadata, and synthetic schemas.

Model- and protocol-specific checks are added in staged revision rounds.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_synthetic_data import write_synthetic_inputs  # noqa: E402
from src.config import CONFIG_DIR, load_config, load_wells_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    wells = load_wells_config()
    assert cfg["seed"]["global"] == 20260827
    assert cfg["uft"]["total_dimensions"] == 50
    assert wells.totals["total_samples"] == 1259
    assert wells.totals["blind_samples"] == 369

    # Every paper-facing YAML must parse.
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            yaml.safe_load(f)

    # Machine-readable feature dictionary must contain exactly 50 features.
    feature_dict = REPO_ROOT / "metadata" / "feature_dictionary.csv"
    with open(feature_dict, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 50
    assert [int(r["feature_index"]) for r in rows] == list(range(50))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        write_synthetic_inputs(out, seed=20260827)
        logs = pd.read_csv(out / "continuous_logs.csv", comment="#")
        core = pd.read_csv(out / "core_samples.csv", comment="#")
        lith = pd.read_csv(out / "lithology_corpus.csv", comment="#")
        assert len(core) == 1259
        assert set(["GR", "CNL", "DEN", "AC", "RLA5"]).issubset(logs.columns)
        assert set(["sample_id", "porosity_pct", "permeability_mD"]).issubset(core.columns)
        assert set(["lithology_class_id", "source_sample_id"]).issubset(lith.columns)
        assert lith["lithology_class_id"].between(0, 17).all()

    print("Foundation smoke test passed: configs, metadata, and synthetic schemas are valid.")


if __name__ == "__main__":
    main()
