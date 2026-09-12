from pathlib import Path

import numpy as np

from scripts.generate_synthetic_data import write_synthetic_inputs
from src.analysis.label_sensitivity import cluster_validity_scan, kmeans_definition
from src.pipeline import build_context


ROOT = Path(__file__).resolve().parents[1]


def test_table8a_scan_has_c2_to_c6(tmp_path):
    write_synthetic_inputs(tmp_path, 20260827)
    ctx = build_context(tmp_path)
    scan = cluster_validity_scan(ctx.development)
    assert scan["n_classes"].tolist() == [2, 3, 4, 5, 6]
    assert np.isfinite(scan[["silhouette", "calinski_harabasz", "davies_bouldin"]]).all().all()


def test_alternative_kmeans_definitions_cover_all_rows(tmp_path):
    write_synthetic_inputs(tmp_path, 20260827)
    ctx = build_context(tmp_path)
    full = ctx.labeled_core
    devmask = full["well_id"].isin(ctx.wells.training_well_names).to_numpy()
    for c in (2, 3, 4):
        d = kmeans_definition(full, devmask, n_classes=c)
        assert d.labels_all.shape == (len(full),)
        assert set(np.unique(d.labels_all)) == set(range(c))
        assert len(d.boundaries) == c - 1
