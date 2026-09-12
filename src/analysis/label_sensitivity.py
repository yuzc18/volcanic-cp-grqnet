"""Label-definition sensitivity utilities for manuscript Table 8.

The class-count scan is fitted on the five-well 890-sample development set in
log10(FZI) space. Alternative boundaries are likewise fitted on those development
rows and then frozen before they are applied to all seven wells.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.data.labels import compute_fzi, kmedoids_alternative_labels


@dataclass(frozen=True)
class LabelDefinition:
    name: str
    n_classes: int
    labels_all: np.ndarray
    boundaries: np.ndarray


def cluster_validity_scan(
    development: pd.DataFrame,
    *,
    class_counts: tuple[int, ...] = (2, 3, 4, 5, 6),
    random_state: int = 42,
) -> pd.DataFrame:
    """Return the Table-8(a) internal cluster-validity scan."""
    fzi = compute_fzi(development["porosity_pct"], development["permeability_mD"])
    x = np.log10(fzi).reshape(-1, 1)
    rows = []
    for c in class_counts:
        km = KMeans(
            n_clusters=int(c), init="k-means++", n_init=10,
            max_iter=300, tol=1e-4, random_state=int(random_state),
        ).fit(x)
        lab = km.labels_.astype(int)
        centers = np.sort(km.cluster_centers_.ravel())
        boundaries = 10.0 ** (0.5 * (centers[:-1] + centers[1:]))
        counts = np.bincount(lab, minlength=int(c))
        rows.append({
            "n_classes": int(c),
            "silhouette": float(silhouette_score(x, lab)),
            "calinski_harabasz": float(calinski_harabasz_score(x, lab)),
            "davies_bouldin": float(davies_bouldin_score(x, lab)),
            "smallest_class_n": int(counts.min()),
            "boundaries_um": " / ".join(f"{v:.6g}" for v in boundaries),
        })
    return pd.DataFrame(rows)


def _labels_from_boundaries(fzi: np.ndarray, boundaries: np.ndarray, n_classes: int) -> np.ndarray:
    # searchsorted(..., side='right') assigns an exact boundary to the
    # higher-FZI class after the high-to-low ID reversal.
    low_to_high = np.searchsorted(np.asarray(boundaries, dtype=float), fzi, side="right")
    return (int(n_classes) - 1 - low_to_high).astype(int)


def kmeans_definition(
    full_rows: pd.DataFrame,
    development_mask: np.ndarray,
    *,
    n_classes: int,
    random_state: int = 42,
) -> LabelDefinition:
    if n_classes not in {2, 3, 4}:
        raise ValueError("Table-8 K-means definitions use c=2, c=3, or c=4.")
    dev = full_rows.loc[np.asarray(development_mask, dtype=bool)]
    fzi_dev = compute_fzi(dev["porosity_pct"], dev["permeability_mD"])
    x = np.log10(fzi_dev).reshape(-1, 1)
    km = KMeans(
        n_clusters=int(n_classes), init="k-means++", n_init=10,
        max_iter=300, tol=1e-4, random_state=int(random_state),
    ).fit(x)
    centers = np.sort(km.cluster_centers_.ravel())
    boundaries = 10.0 ** (0.5 * (centers[:-1] + centers[1:]))
    fzi_all = compute_fzi(full_rows["porosity_pct"], full_rows["permeability_mD"])
    labels_all = _labels_from_boundaries(fzi_all, boundaries, n_classes)
    return LabelDefinition(f"kmeans_c{n_classes}", n_classes, labels_all, boundaries)


def kmedoids_definition(
    full_rows: pd.DataFrame,
    development_mask: np.ndarray,
    *,
    random_state: int = 42,
) -> LabelDefinition:
    dev = full_rows.loc[np.asarray(development_mask, dtype=bool)]
    _, boundaries = kmedoids_alternative_labels(
        dev["porosity_pct"], dev["permeability_mD"], random_state=random_state
    )
    fzi_all = compute_fzi(full_rows["porosity_pct"], full_rows["permeability_mD"])
    labels_all = _labels_from_boundaries(fzi_all, boundaries, 3)
    return LabelDefinition("kmedoids_l1_c3", 3, labels_all, np.asarray(boundaries, dtype=float))
