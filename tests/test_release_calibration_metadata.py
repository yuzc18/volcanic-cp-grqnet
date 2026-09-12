from pathlib import Path

import pandas as pd

from scripts.generate_synthetic_data import write_synthetic_inputs
from src.pipeline import build_context

ROOT = Path(__file__).resolve().parents[1]


def test_release_does_not_ship_reconstructed_row_level_split_manifest():
    assert not (ROOT / "metadata" / "splits.csv").exists()
    scan_roots = [ROOT / "metadata", ROOT / "configs", ROOT / "src", ROOT / "scripts", ROOT / "README.md", ROOT / "CODE_AVAILABILITY.md", ROOT / "MANUSCRIPT_CODE_ALIGNMENT.md"]
    files = []
    for base in scan_roots:
        if base.is_file():
            files.append(base)
        else:
            files.extend(p for p in base.rglob("*") if p.is_file())
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in files
        if p.suffix.lower() in {".md", ".csv", ".yaml", ".yml", ".py", ".txt"}
    )
    forbidden = "Table3b_" + "reconstructed_anonymous_slot"
    assert forbidden not in text


def test_released_aggregate_calibration_metadata_matches_table3b():
    f = pd.read_csv(ROOT / "metadata" / "fold_assignments.csv")
    assert f["calibration_n"].tolist() == [32, 28, 52, 31, 35]
    assert int(f["calibration_n"].sum()) == 178
    assert [int(f[c].sum()) for c in ["calibration_H", "calibration_M", "calibration_L"]] == [51, 49, 78]
    assert f["validation_n"].tolist() == [128, 114, 206, 125, 139]
    assert f["outer_training_n"].tolist() == [584, 598, 506, 587, 573]


def test_optional_authoritative_id_file_roundtrips(tmp_path):
    data_dir = tmp_path / "data"
    write_synthetic_inputs(data_dir, 20260827)
    base = build_context(data_dir, cp_seed=20260827)
    dev = base.partition.development_df
    ids = dev.iloc[base.partition.calibration.cp_calib_idx]["sample_id"].astype(str).tolist()
    meta = pd.DataFrame({
        "sample_id": dev["sample_id"].astype(str),
        "is_main_calibration": dev["sample_id"].astype(str).isin(ids),
    })
    path = tmp_path / "authoritative_split.csv"
    meta.to_csv(path, index=False)
    rebuilt = build_context(data_dir, cp_seed=999, split_metadata_path=path)
    assert set(rebuilt.calibration_sample_ids) == set(ids)
    assert len(rebuilt.calibration_sample_ids) == 178
