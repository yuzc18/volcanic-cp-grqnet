import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_release_contains_export_contract_not_table3a_copies():
    d = ROOT / "metadata" / "per_fold_configs"
    assert (d / "README.md").exists()
    assert (d / "schema.json").exists()
    assert (d / "manifest.yaml").exists()
    assert not (d / "study_reference.yaml").exists()
    assert not (d / "per_fold_configurations.csv").exists()
    text = "\n".join(p.read_text(encoding="utf-8") for p in d.iterdir() if p.suffix in {".md", ".json", ".yaml"})
    assert "Table_3a_reference_reconstruction" not in text
    assert "manuscript_consistent_reconstruction_from_Table_3a" not in text
    manifest = yaml.safe_load((d / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["source_of_selected_values"] == "actual_fold_specific_hpo_run"
    assert [x["validation_well"] for x in manifest["outer_folds"]] == ["WF1", "CS12", "CS602", "CS607", "CS11"]
    assert manifest["final_model"]["tuning_subset_n"] == 712
    assert manifest["final_model"]["refit_on_all_712"] is True


def test_export_schema_requires_real_selection_fields():
    schema = json.loads((ROOT / "metadata" / "per_fold_configs" / "schema.json").read_text(encoding="utf-8"))
    fields = schema["fields"]
    for name in ["model", "fold", "validation_well", "selection_engine", "selection_metric", "internal_macro_f1", "selected_params", "seed"]:
        assert name in fields
