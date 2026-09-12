"""Randomized adaptive prediction sets (APS) for the finalized manuscript.

The implementation follows the manuscript's split-conformal construction:

* randomized APS calibration score
  ``s = cumulative_to_true - U * p_true``;
* finite-sample threshold equal to the
  ``ceil((n+1)(1-alpha))``-th ordered calibration score;
* prediction classes accumulated in descending probability order until the
  cumulative probability first exceeds ``q_hat``;
* randomized removal of that boundary class with probability
  ``(cum_boundary - q_hat) / p_boundary``;
* the top-ranked class is always retained, so prediction sets are non-empty.

The public repository contains no study data.  The same code therefore runs on
synthetic inputs, whose numerical outputs are not expected to reproduce the
manuscript values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_probabilities(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("probs must have shape (n_samples, n_classes>=2).")
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("probs must be finite and non-negative.")
    sums = arr.sum(axis=1)
    if (sums <= 0).any():
        raise ValueError("each probability row must have positive mass.")
    # Softmax outputs should already sum to one.  Normalizing here makes the
    # low-level function robust to harmless floating-point drift only.
    return arr / sums[:, None]


def _uniforms(
    n: int,
    *,
    rng: np.random.Generator | None = None,
    uniforms: np.ndarray | None = None,
) -> np.ndarray:
    if uniforms is not None:
        u = np.asarray(uniforms, dtype=float)
        if u.shape != (n,):
            raise ValueError(f"uniforms must have shape ({n},).")
        if not np.isfinite(u).all() or (u < 0).any() or (u > 1).any():
            raise ValueError("uniforms must lie in [0, 1].")
        return u
    if rng is None:
        raise ValueError("provide either rng or uniforms for randomized APS.")
    return rng.uniform(0.0, 1.0, size=n)


def aps_nonconformity_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    uniforms: np.ndarray | None = None,
    randomized: bool = True,
) -> np.ndarray:
    """Compute randomized APS calibration scores.

    ``labels`` are integer class IDs.  With ``randomized=False`` the uniform
    term is set to zero, yielding the conservative cumulative-probability score.
    """
    p = _validate_probabilities(probs)
    y = np.asarray(labels, dtype=int)
    n, k = p.shape
    if y.shape != (n,):
        raise ValueError("labels length must match probability rows.")
    if (y < 0).any() or (y >= k).any():
        raise ValueError("labels contain class IDs outside probability columns.")

    u = _uniforms(n, rng=rng, uniforms=uniforms) if randomized else np.zeros(n)
    order = np.argsort(-p, axis=1, kind="stable")
    scores = np.empty(n, dtype=float)
    for i in range(n):
        cum = 0.0
        true = int(y[i])
        for cls in order[i]:
            cls = int(cls)
            cum += float(p[i, cls])
            if cls == true:
                scores[i] = cum - float(u[i]) * float(p[i, true])
                break
    return scores


def finite_sample_order_statistic(n: int, alpha: float) -> int:
    """Return the 1-based finite-sample order statistic index.

    For the manuscript primary design, ``n=178`` and ``alpha=0.05`` give 171.
    If the nominal order exceeds ``n`` (very small alpha), it is capped at n.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    return min(int(np.ceil((n + 1) * (1.0 - alpha))), n)


def compute_threshold(scores: np.ndarray, alpha: float) -> float:
    """Return the exact finite-sample APS threshold as an ordered score."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or len(s) == 0 or not np.isfinite(s).all():
        raise ValueError("scores must be a non-empty finite 1-D array.")
    kth = finite_sample_order_statistic(len(s), alpha)
    return float(np.sort(s, kind="stable")[kth - 1])


@dataclass(frozen=True)
class PredictionSetDetails:
    sets: list[list[int]]
    sizes: np.ndarray
    boundary_removed: np.ndarray
    uniforms: np.ndarray


def build_prediction_sets_detailed(
    probs: np.ndarray,
    q_hat: float,
    *,
    rng: np.random.Generator | None = None,
    uniforms: np.ndarray | None = None,
    randomized: bool = True,
    always_retain_top: bool = True,
) -> PredictionSetDetails:
    """Construct randomized, non-empty APS prediction sets.

    Let ``k*`` be the first rank where cumulative probability exceeds q_hat.
    The boundary class is removed with probability
    ``(cum_k* - q_hat) / p_k*``.  The highest-probability class is retained even
    when the randomized rule would remove it.
    """
    p = _validate_probabilities(probs)
    m, k = p.shape
    if not np.isfinite(q_hat):
        raise ValueError("q_hat must be finite.")
    u = _uniforms(m, rng=rng, uniforms=uniforms) if randomized else np.ones(m)
    order = np.argsort(-p, axis=1, kind="stable")

    sets: list[list[int]] = []
    sizes = np.empty(m, dtype=np.int64)
    removed = np.zeros(m, dtype=bool)

    for i in range(m):
        cum = 0.0
        current: list[int] = []
        boundary_prob = 0.0
        boundary_rank = 0
        for rank, cls0 in enumerate(order[i]):
            cls = int(cls0)
            boundary_rank = rank
            boundary_prob = float(p[i, cls])
            cum += boundary_prob
            current.append(cls)
            # The manuscript uses the first rank whose cumulative mass strictly
            # exceeds q_hat; the final-rank guard handles q_hat at the simplex edge.
            if cum > q_hat or rank == k - 1:
                break

        if randomized and boundary_prob > 0:
            removal_prob = float(np.clip((cum - q_hat) / boundary_prob, 0.0, 1.0))
            remove = float(u[i]) < removal_prob
            if remove and not (always_retain_top and boundary_rank == 0):
                current.pop()
                removed[i] = True

        if always_retain_top and not current:
            current = [int(order[i, 0])]
            removed[i] = False

        sets.append(current)
        sizes[i] = len(current)

    return PredictionSetDetails(sets=sets, sizes=sizes, boundary_removed=removed, uniforms=u)


def build_prediction_sets(
    probs: np.ndarray,
    q_hat: float,
    *,
    rng: np.random.Generator | None = None,
    uniforms: np.ndarray | None = None,
    randomized: bool = True,
    always_retain_top: bool = True,
) -> tuple[list[list[int]], np.ndarray]:
    """Backward-compatible convenience wrapper returning sets and sizes."""
    out = build_prediction_sets_detailed(
        probs,
        q_hat,
        rng=rng,
        uniforms=uniforms,
        randomized=randomized,
        always_retain_top=always_retain_top,
    )
    return out.sets, out.sizes


@dataclass(frozen=True)
class CoverageReport:
    marginal_coverage: float
    mean_set_size: float
    singleton_rate: float
    set_size_distribution: dict[int, int]
    per_class_coverage: dict[int, float]


class ConformalAPS:
    """Stateful randomized APS predictor with one fixed random-number stream.

    Calling :meth:`calibrate` resets the stream to ``seed``, draws the
    calibration uniforms once, and stores the scores/threshold.  Subsequent
    calls to :meth:`predict` draw boundary uniforms from the same stream.  For
    analyses that must reuse exactly the same random draws across several alpha
    levels, use the low-level functions with explicit ``uniforms`` arrays.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        seed: int = 20260827,
        randomized: bool = True,
        always_retain_top: bool = True,
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.randomized = bool(randomized)
        self.always_retain_top = bool(always_retain_top)
        self.q_hat: float | None = None
        self.calibration_scores_: np.ndarray | None = None
        self.calibration_uniforms_: np.ndarray | None = None
        self._rng: np.random.Generator | None = None

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> "ConformalAPS":
        self._rng = np.random.default_rng(self.seed)
        n = len(labels)
        u = self._rng.uniform(0.0, 1.0, size=n) if self.randomized else np.zeros(n)
        scores = aps_nonconformity_scores(
            probs, labels, uniforms=u, randomized=self.randomized
        )
        self.calibration_uniforms_ = u
        self.calibration_scores_ = scores
        self.q_hat = compute_threshold(scores, self.alpha)
        return self

    def predict(
        self,
        probs: np.ndarray,
        *,
        uniforms: np.ndarray | None = None,
    ) -> tuple[list[list[int]], np.ndarray]:
        if self.q_hat is None:
            raise RuntimeError("ConformalAPS must be calibrated before predict().")
        if uniforms is None and self.randomized:
            if self._rng is None:
                raise RuntimeError("random-number stream is not initialized.")
            uniforms = self._rng.uniform(0.0, 1.0, size=len(probs))
        return build_prediction_sets(
            probs,
            self.q_hat,
            uniforms=uniforms,
            randomized=self.randomized,
            always_retain_top=self.always_retain_top,
        )

    def evaluate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        *,
        n_classes: int = 3,
        uniforms: np.ndarray | None = None,
    ) -> CoverageReport:
        sets, sizes = self.predict(probs, uniforms=uniforms)
        y = np.asarray(labels, dtype=int)
        covered = np.array([int(y[i]) in sets[i] for i in range(len(y))], dtype=bool)
        size_dist = {int(s): int(np.sum(sizes == s)) for s in np.unique(sizes)}
        per_class = {
            c: float(covered[y == c].mean()) if np.any(y == c) else float("nan")
            for c in range(n_classes)
        }
        return CoverageReport(
            marginal_coverage=float(covered.mean()),
            mean_set_size=float(sizes.mean()),
            singleton_rate=float(np.mean(sizes == 1)),
            set_size_distribution=size_dist,
            per_class_coverage=per_class,
        )
