"""FZI-derived reservoir-quality labels used by the revision.

The implementation follows manuscript Section 3.2:

* RQI = 0.0314 * sqrt(K / phi), with K in mD and phi as a fraction;
* phi_z = phi / (1 - phi);
* FZI = RQI / phi_z, in micrometres;
* K-means is fitted once to log10(FZI) from the five development wells;
* adjacent centres are bisected in log10 space and back-transformed;
* those two FZI boundaries are frozen for all downstream evaluations;
* a sample exactly on a boundary is assigned to the higher-quality class.

The module does not alter recorded porosity values, including records affected
by the source-report 4.00% convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

CLASS_HIGH = 0
CLASS_MEDIUM = 1
CLASS_LOW = 2
CLASS_NAMES = ["High", "Medium", "Low"]
RQI_CONSTANT = 0.0314


def compute_fzi(
    porosity_pct: np.ndarray | pd.Series | Iterable[float],
    permeability_mD: np.ndarray | pd.Series | Iterable[float],
) -> np.ndarray:
    """Compute FZI from recorded porosity (%) and permeability (mD)."""
    phi = np.asarray(porosity_pct, dtype=float) / 100.0
    k = np.asarray(permeability_mD, dtype=float)
    if phi.shape != k.shape:
        raise ValueError("porosity and permeability must have the same shape.")
    if not np.isfinite(phi).all() or not np.isfinite(k).all():
        raise ValueError("FZI inputs must be finite.")
    if (phi <= 0).any() or (phi >= 1).any():
        raise ValueError("porosity must be strictly between 0 and 100 percent.")
    if (k <= 0).any():
        raise ValueError("permeability must be strictly positive.")
    rqi = RQI_CONSTANT * np.sqrt(k / phi)
    phi_z = phi / (1.0 - phi)
    return rqi / phi_z


@dataclass(frozen=True)
class FZIBoundaries:
    """Frozen lower and upper FZI boundaries, in micrometres."""

    medium_low: float
    high_medium: float

    def __post_init__(self) -> None:
        if not (0 < self.medium_low < self.high_medium):
            raise ValueError("Expected 0 < medium_low < high_medium.")

    def as_array(self) -> np.ndarray:
        return np.array([self.medium_low, self.high_medium], dtype=float)


@dataclass(frozen=True)
class FZILabelFit:
    labels: np.ndarray
    fzi: np.ndarray
    log10_fzi: np.ndarray
    centers_log10_desc: np.ndarray
    boundaries: FZIBoundaries
    kmeans: KMeans


class FZILabeler:
    """Fit the development-set label rule once, then apply frozen boundaries."""

    def __init__(
        self,
        *,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.fit_: FZILabelFit | None = None

    def fit(
        self,
        porosity_pct: np.ndarray | pd.Series,
        permeability_mD: np.ndarray | pd.Series,
    ) -> "FZILabeler":
        fzi = compute_fzi(porosity_pct, permeability_mD)
        log10_fzi = np.log10(fzi)
        km = KMeans(
            n_clusters=3,
            init="k-means++",
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        km.fit(log10_fzi.reshape(-1, 1))
        centers_desc = np.sort(km.cluster_centers_.ravel())[::-1]
        high_medium_log = 0.5 * (centers_desc[0] + centers_desc[1])
        medium_low_log = 0.5 * (centers_desc[1] + centers_desc[2])
        boundaries = FZIBoundaries(
            medium_low=float(10.0 ** medium_low_log),
            high_medium=float(10.0 ** high_medium_log),
        )
        labels = assign_from_fzi(fzi, boundaries)
        self.fit_ = FZILabelFit(
            labels=labels,
            fzi=fzi,
            log10_fzi=log10_fzi,
            centers_log10_desc=centers_desc,
            boundaries=boundaries,
            kmeans=km,
        )
        return self

    @property
    def boundaries_(self) -> FZIBoundaries:
        if self.fit_ is None:
            raise RuntimeError("FZILabeler must be fit first.")
        return self.fit_.boundaries

    def predict_fzi(self, fzi: np.ndarray | pd.Series) -> np.ndarray:
        return assign_from_fzi(np.asarray(fzi, dtype=float), self.boundaries_)

    def predict(
        self,
        porosity_pct: np.ndarray | pd.Series,
        permeability_mD: np.ndarray | pd.Series,
    ) -> np.ndarray:
        return self.predict_fzi(compute_fzi(porosity_pct, permeability_mD))


# Manuscript display/acceptance values. They are not imposed on synthetic data.
MANUSCRIPT_BOUNDARIES = FZIBoundaries(medium_low=0.7018, high_medium=1.0549)


def assign_from_fzi(fzi: np.ndarray, boundaries: FZIBoundaries) -> np.ndarray:
    """Apply frozen boundaries with ties assigned to the higher-quality class."""
    fzi = np.asarray(fzi, dtype=float)
    if not np.isfinite(fzi).all() or (fzi <= 0).any():
        raise ValueError("FZI values must be finite and positive.")
    labels = np.full(fzi.shape, CLASS_LOW, dtype=np.int64)
    # Equality at the lower boundary belongs to Medium; equality at the upper
    # boundary belongs to High, i.e. the higher-quality side in both cases.
    labels[fzi >= boundaries.medium_low] = CLASS_MEDIUM
    labels[fzi >= boundaries.high_medium] = CLASS_HIGH
    return labels


def fit_development_labels(
    core_samples: pd.DataFrame,
    development_wells: Iterable[str],
    *,
    labeler: FZILabeler | None = None,
) -> tuple[FZILabeler, pd.DataFrame]:
    """Fit on development wells only and append FZI/quality labels to all rows."""
    required = {"well_id", "porosity_pct", "permeability_mD"}
    missing = required - set(core_samples.columns)
    if missing:
        raise ValueError(f"core_samples missing columns: {sorted(missing)}")
    dev_mask = core_samples["well_id"].isin(list(development_wells)).to_numpy()
    if not dev_mask.any():
        raise ValueError("No development rows selected for FZI clustering.")
    labeler = labeler or FZILabeler()
    labeler.fit(
        core_samples.loc[dev_mask, "porosity_pct"],
        core_samples.loc[dev_mask, "permeability_mD"],
    )
    out = core_samples.copy()
    out["FZI_um"] = compute_fzi(out["porosity_pct"], out["permeability_mD"])
    out["quality_class_id"] = labeler.predict_fzi(out["FZI_um"].to_numpy())
    out["quality_class"] = [CLASS_NAMES[i] for i in out["quality_class_id"]]
    return labeler, out


def label_stability_score(
    porosity_pct: np.ndarray | pd.Series,
    permeability_mD: np.ndarray | pd.Series,
    reference_labels: np.ndarray,
    *,
    n_repeats: int = 50,
    base_seed: int = 42,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> np.ndarray:
    """Repeat single-start k-means++ and Hungarian-match to adopted labels.

    Each repeat uses ``n_init=1`` exactly as specified in the revision. Seeds
    are distinct and deterministic: ``base_seed, ..., base_seed+n_repeats-1``.
    """
    fzi = compute_fzi(porosity_pct, permeability_mD)
    x = np.log10(fzi).reshape(-1, 1)
    ref = np.asarray(reference_labels, dtype=int)
    if len(ref) != len(x):
        raise ValueError("reference_labels length mismatch.")
    agree = np.zeros(len(ref), dtype=int)
    for i in range(n_repeats):
        km = KMeans(
            n_clusters=3,
            init="k-means++",
            n_init=1,
            max_iter=max_iter,
            tol=tol,
            random_state=base_seed + i,
        )
        raw = km.fit_predict(x)
        counts = np.zeros((3, 3), dtype=int)
        for r in range(3):
            for c in range(3):
                counts[r, c] = int(np.sum((raw == r) & (ref == c)))
        rows, cols = linear_sum_assignment(-counts)
        mapping = np.empty(3, dtype=int)
        mapping[rows] = cols
        agree += (mapping[raw] == ref)
    return agree / float(n_repeats)


def bootstrap_boundaries(
    porosity_pct: np.ndarray | pd.Series,
    permeability_mD: np.ndarray | pd.Series,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 20260827,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> np.ndarray:
    """IID bootstrap of the two FZI boundaries; refit K-means every replicate."""
    por = np.asarray(porosity_pct, dtype=float)
    perm = np.asarray(permeability_mD, dtype=float)
    if len(por) != len(perm):
        raise ValueError("Input lengths differ.")
    rng = np.random.default_rng(random_state)
    out = np.empty((n_bootstrap, 2), dtype=float)
    n = len(por)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        lab = FZILabeler(
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=42,
        ).fit(por[idx], perm[idx])
        out[b] = lab.boundaries_.as_array()
    return out


def kmeans_alternative_boundaries(
    porosity_pct: np.ndarray | pd.Series,
    permeability_mD: np.ndarray | pd.Series,
    *,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct c=2 or c=4 K-means alternatives in log10(FZI) space."""
    if n_clusters not in {2, 4}:
        raise ValueError("The revision's K-means alternatives use c=2 or c=4.")
    fzi = compute_fzi(porosity_pct, permeability_mD)
    x = np.log10(fzi).reshape(-1, 1)
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    ).fit(x)
    centers = np.sort(km.cluster_centers_.ravel())
    boundary_log = 0.5 * (centers[:-1] + centers[1:])
    boundaries = 10.0 ** boundary_log
    labels_low_to_high = np.searchsorted(boundaries, fzi, side="right")
    labels_high_to_low = (n_clusters - 1 - labels_low_to_high).astype(int)
    return labels_high_to_low, boundaries


def kmedoids_alternative_labels(
    porosity_pct: np.ndarray | pd.Series,
    permeability_mD: np.ndarray | pd.Series,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Three-class Manhattan K-medoids alternative using scikit-learn-extra."""
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError as exc:  # pragma: no cover - dependency is pinned for release
        raise ImportError(
            "scikit-learn-extra is required for the K-medoids sensitivity analysis; "
            "install requirements.txt."
        ) from exc

    fzi = compute_fzi(porosity_pct, permeability_mD)
    x = np.log10(fzi).reshape(-1, 1)
    model = KMedoids(
        n_clusters=3,
        metric="manhattan",
        init="k-medoids++",
        random_state=random_state,
    ).fit(x)
    centers = np.asarray(model.cluster_centers_, dtype=float).ravel()
    order = np.argsort(centers)[::-1]
    mapping = np.empty(3, dtype=int)
    mapping[order] = np.arange(3)
    labels = mapping[model.labels_]
    centers_asc = np.sort(centers)
    boundaries = 10.0 ** (0.5 * (centers_asc[:-1] + centers_asc[1:]))
    return labels.astype(int), boundaries


# ---------------------------------------------------------------------------
# Backward-compatible function used by legacy tests/scripts.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelResult:
    labels: np.ndarray
    fzi: np.ndarray
    log_fzi: np.ndarray
    cluster_centers_log_fzi: np.ndarray
    boundaries_log_fzi: np.ndarray


def assign_quality_labels(
    porosity_pct: np.ndarray | pd.Series,
    perm_mD: np.ndarray | pd.Series,
    *,
    seed: int = 42,
    n_clusters: int = 3,
    n_init: int = 10,
    max_iter: int = 300,
) -> LabelResult:
    if n_clusters != 3:
        raise ValueError("Main analysis fixes n_clusters=3.")
    lab = FZILabeler(
        n_init=n_init, max_iter=max_iter, tol=1e-4, random_state=seed
    ).fit(porosity_pct, perm_mD)
    fit = lab.fit_
    assert fit is not None
    b = fit.boundaries
    return LabelResult(
        labels=fit.labels,
        fzi=fit.fzi,
        log_fzi=fit.log10_fzi,
        cluster_centers_log_fzi=fit.centers_log10_desc,
        boundaries_log_fzi=np.log10([b.high_medium, b.medium_low]),
    )
