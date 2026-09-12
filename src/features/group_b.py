"""Group B: 18-dimensional lithology-probability vector.

The revision requires the upstream lithology pipeline to be refitted for every
active data split.  Training-only preprocessing is:

    raw five LWD curves -> z-score -> Borderline-SMOTE -> eRF -> 18 probabilities

Reference [50] defines eRF by a 300-tree ensemble combining C4.5 gain-ratio
splitting, Borderline-SMOTE, and Kendall-concordance feature-stability guidance.
This module implements that published algorithmic specification directly rather
than vendoring the external companion repository.  Reference [50] reports the
high-level deployed settings and candidate search spaces, but does not report
all selected low-level values; those implementation choices are made explicit
in ``configs/erf.yaml`` and are not presented as recovered historical settings.
Probability columns and the refitting/exclusion protocol are paper-facing
contracts.

Round 3 supplies the outer/inner leave-one-well-out orchestration.  Round 2
provides the leakage-safe fit/predict primitive used by that orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler

from src.data.io import LITH_TARGET, LWD_COLUMNS

GROUP_B_COLUMNS = [f"LithProb_{i}" for i in range(18)]
LITHOLOGY_CLASS_NAMES = [
    "Basalt",
    "Andesite",
    "Rhyolite",
    "Andesitic volcanic breccia",
    "Andesitic brecciated lava",
    "Andesitic tuff lava",
    "Andesitic tuff",
    "Rhyolitic brecciated lava",
    "Rhyolitic tuff lava",
    "Rhyolitic volcanic breccia",
    "Rhyolitic tuff",
    "Sedimentary volcanic breccia",
    "Sedimentary tuff",
    "Trachytic brecciated lava",
    "Trachytic tuff lava",
    "Trachytic volcanic breccia",
    "Trachytic tuff",
    "Trachyte",
]
N_LITH_CLASSES = len(LITHOLOGY_CLASS_NAMES)


def _entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    """Entropy for one or many class-count vectors (last axis = classes)."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    logp = np.zeros_like(p)
    mask = p > 0
    logp[mask] = np.log2(p[mask])
    return -(p * logp).sum(axis=-1)


@dataclass
class _Node:
    probs: np.ndarray
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class C45GainRatioTree:
    """Continuous-feature multiclass C4.5-style tree using gain ratio.

    Every distinct adjacent feature value is a candidate split.  A feature
    stability weight supplied by the ensemble multiplies its gain ratio; this
    is how the independent eRF reconstruction lets consensus from preceding
    trees guide subsequent splitting while retaining C4.5's node-level rule.
    """

    def __init__(
        self,
        *,
        n_classes: int = N_LITH_CLASSES,
        min_samples_leaf: int = 1,
        max_depth: int | None = None,
        max_features: str | int | None = "sqrt",
        n_bins: int | None = 16,
        random_state: int = 0,
        feature_weights: np.ndarray | None = None,
    ) -> None:
        self.n_classes = int(n_classes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_bins = None if n_bins is None else int(n_bins)
        self.random_state = int(random_state)
        self.feature_weights = feature_weights
        self.root_: _Node | None = None
        self.feature_importances_: np.ndarray | None = None
        self._importance_raw: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "C45GainRatioTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("Invalid X/y shapes for C45GainRatioTree.")
        if len(X) == 0:
            raise ValueError("Cannot fit an empty tree.")
        n_features = X.shape[1]
        if self.feature_weights is None:
            self.feature_weights = np.ones(n_features, dtype=float)
        else:
            self.feature_weights = np.asarray(self.feature_weights, dtype=float)
            if self.feature_weights.shape != (n_features,):
                raise ValueError("feature_weights shape mismatch.")
        self._importance_raw = np.zeros(n_features, dtype=float)
        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow(X, y, depth=0)
        total = float(self._importance_raw.sum())
        self.feature_importances_ = (
            self._importance_raw / total if total > 0 else np.zeros(n_features, dtype=float)
        )
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        probs = counts / counts.sum()
        node = _Node(probs=probs)
        if np.count_nonzero(counts) <= 1:
            return node
        if len(y) < 2 * self.min_samples_leaf:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        best = self._best_split(X, y)
        if best is None:
            return node
        feature, threshold, raw_gain_ratio = best
        mask = X[:, feature] <= threshold
        n_left = int(mask.sum())
        if n_left < self.min_samples_leaf or (len(y) - n_left) < self.min_samples_leaf:
            return node

        node.feature = feature
        node.threshold = threshold
        assert self._importance_raw is not None
        self._importance_raw[feature] += raw_gain_ratio * len(y)
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def _feature_subset(self, n_features: int) -> np.ndarray:
        mf = self.max_features
        if mf is None:
            k = n_features
        elif isinstance(mf, int):
            k = max(1, min(int(mf), n_features))
        elif mf == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif mf == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Unsupported max_features={mf!r}")
        if k >= n_features:
            return np.arange(n_features, dtype=int)
        return np.sort(self._rng.choice(n_features, size=k, replace=False))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float, float] | None:
        n = len(y)
        parent_counts = np.bincount(y, minlength=self.n_classes)
        parent_entropy = float(_entropy_from_counts(parent_counts))
        if parent_entropy <= 0:
            return None

        best_score = -np.inf
        best: tuple[int, float, float] | None = None
        all_classes = np.arange(self.n_classes)

        for j in self._feature_subset(X.shape[1]):
            order = np.argsort(X[:, j], kind="mergesort")
            xs = X[order, j]
            ys = y[order]
            # Candidate after index i means left uses [0..i], right [i+1..].
            valid_change = xs[:-1] < xs[1:]
            positions = np.nonzero(valid_change)[0]
            positions = positions[
                (positions + 1 >= self.min_samples_leaf)
                & (n - positions - 1 >= self.min_samples_leaf)
            ]
            if len(positions) == 0:
                continue
            if self.n_bins is not None and self.n_bins > 0 and len(positions) > self.n_bins:
                # C4.5-style continuous threshold search evaluated on a bounded
                # set of ordered candidate locations (published eRF n_bins).
                take = np.unique(np.rint(np.linspace(0, len(positions) - 1, self.n_bins)).astype(int))
                positions = positions[take]

            onehot = (ys[:, None] == all_classes[None, :]).astype(np.int32)
            cum = np.cumsum(onehot, axis=0)
            left_counts = cum[positions]
            right_counts = parent_counts[None, :] - left_counts
            n_left = positions + 1
            n_right = n - n_left
            h_left = _entropy_from_counts(left_counts)
            h_right = _entropy_from_counts(right_counts)
            info_gain = parent_entropy - (n_left / n) * h_left - (n_right / n) * h_right
            split_counts = np.column_stack([n_left, n_right])
            split_info = _entropy_from_counts(split_counts)
            ratio = np.divide(
                info_gain,
                split_info,
                out=np.zeros_like(info_gain, dtype=float),
                where=split_info > 1e-15,
            )
            idx = int(np.argmax(ratio))
            raw_ratio = float(ratio[idx])
            weighted = raw_ratio * float(self.feature_weights[j])
            if weighted > best_score and raw_ratio > 1e-15:
                pos = int(positions[idx])
                threshold = 0.5 * (float(xs[pos]) + float(xs[pos + 1]))
                best_score = weighted
                best = (j, threshold, raw_ratio)
        return best

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Tree must be fit before predict_proba.")
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), self.n_classes), dtype=float)
        for i, row in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                assert node.feature is not None and node.threshold is not None
                node = node.left if row[node.feature] <= node.threshold else node.right  # type: ignore[assignment]
                assert node is not None
            out[i] = node.probs
        return out


class KendallGuidedC45Forest:
    """Bootstrap ensemble with sequential Kendall-consensus feature guidance."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        n_classes: int = N_LITH_CLASSES,
        min_samples_leaf: int = 1,
        max_depth: int | None = None,
        max_features: str | int | None = "sqrt",
        n_bins: int | None = 16,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.n_classes = int(n_classes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_bins = None if n_bins is None else int(n_bins)
        self.random_state = int(random_state)
        self.trees_: list[C45GainRatioTree] = []
        self.rank_history_: list[np.ndarray] = []
        self.kendall_w_: float | None = None
        self.feature_importances_: np.ndarray | None = None

    @staticmethod
    def _ranks_desc(values: np.ndarray) -> np.ndarray:
        order = np.argsort(-values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1, dtype=float)
        return ranks

    @staticmethod
    def _kendall_w(rank_matrix: np.ndarray) -> float:
        """Kendall's coefficient of concordance for tree-wise feature ranks."""
        ranks = np.asarray(rank_matrix, dtype=float)
        m, n = ranks.shape
        if m < 2 or n < 2:
            return 0.0
        rank_sums = ranks.sum(axis=0)
        s = float(((rank_sums - rank_sums.mean()) ** 2).sum())
        denom = m * m * (n**3 - n)
        return float(12.0 * s / denom) if denom > 0 else 0.0

    def _consensus_weights(self, n_features: int) -> np.ndarray:
        if not self.rank_history_:
            return np.ones(n_features, dtype=float)
        ranks = np.vstack(self.rank_history_)
        mean_rank = ranks.mean(axis=0)
        # Consistently high-ranked features receive larger weights.  Mean is
        # normalized to one, preserving the numerical scale of gain ratios.
        scores = (n_features + 1.0) - mean_rank
        scores = np.maximum(scores, 1e-6)
        return scores / scores.mean()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KendallGuidedC45Forest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.rank_history_ = []
        n, p = X.shape
        for t in range(self.n_estimators):
            boot = rng.integers(0, n, size=n)
            weights = self._consensus_weights(p)
            tree = C45GainRatioTree(
                n_classes=self.n_classes,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                max_features=self.max_features,
                n_bins=self.n_bins,
                random_state=self.random_state + t,
                feature_weights=weights,
            ).fit(X[boot], y[boot])
            self.trees_.append(tree)
            assert tree.feature_importances_ is not None
            self.rank_history_.append(self._ranks_desc(tree.feature_importances_))
        ranks = np.vstack(self.rank_history_)
        self.kendall_w_ = self._kendall_w(ranks)
        importance = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        total = float(importance.sum())
        self.feature_importances_ = importance / total if total > 0 else importance
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("Forest must be fit before predict_proba.")
        probs = np.mean([tree.predict_proba(X) for tree in self.trees_], axis=0)
        row_sum = probs.sum(axis=1, keepdims=True)
        return np.divide(probs, row_sum, out=np.full_like(probs, 1 / self.n_classes), where=row_sum > 0)


class ERFLithologyClassifier:
    """Complete Group-B fitting pipeline: z-score + Borderline-SMOTE + eRF."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        random_state: int = 42,
        smote_kwargs: dict | None = None,
        min_samples_leaf: int = 1,
        max_depth: int | None = None,
        max_features: str | int | None = "sqrt",
        n_bins: int | None = 16,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.smote_kwargs = dict(smote_kwargs or {})
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_bins = None if n_bins is None else int(n_bins)
        self.scaler = StandardScaler()
        self.model = KendallGuidedC45Forest(
            n_estimators=n_estimators,
            n_classes=N_LITH_CLASSES,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features=max_features,
            n_bins=n_bins,
            random_state=random_state,
        )
        self.fitted_ = False
        self.classes_seen_: np.ndarray | None = None
        self.smote_applied_: bool = False

    def fit(self, corpus: pd.DataFrame) -> "ERFLithologyClassifier":
        required = {*LWD_COLUMNS, LITH_TARGET}
        missing = required - set(corpus.columns)
        if missing:
            raise ValueError(f"eRF corpus missing columns: {sorted(missing)}")
        X = corpus[LWD_COLUMNS].to_numpy(dtype=float)
        y = corpus[LITH_TARGET].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("eRF training requires at least two lithology classes.")
        Xz = self.scaler.fit_transform(X)

        # Borderline-SMOTE is training-only.  For a tiny synthetic/test subset,
        # some classes can have too few members for the library defaults; adapt
        # neighbours downward without using any evaluation data.
        counts = np.bincount(y, minlength=N_LITH_CLASSES)
        positive = counts[counts > 0]
        min_count = int(positive.min())
        kwargs = dict(self.smote_kwargs)
        if "k_neighbors" not in kwargs:
            kwargs["k_neighbors"] = max(1, min(5, min_count - 1))
        if "m_neighbors" not in kwargs:
            kwargs["m_neighbors"] = max(2, min(10, len(y) - 1))
        kwargs.setdefault("kind", "borderline-1")
        smote = BorderlineSMOTE(random_state=self.random_state, **kwargs)
        try:
            X_fit, y_fit = smote.fit_resample(Xz, y)
            self.smote_applied_ = True
        except ValueError:
            # Borderline-SMOTE can legitimately generate no samples if a demo
            # dataset has no minority borderline points.  The training-only
            # call was still attempted; retain the original standardized rows.
            X_fit, y_fit = Xz, y
            self.smote_applied_ = False

        self.model.fit(X_fit, y_fit)
        self.classes_seen_ = np.unique(y)
        self.fitted_ = True
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("ERFLithologyClassifier must be fit before predict_proba.")
        X = rows[LWD_COLUMNS].to_numpy(dtype=float)
        probs = self.model.predict_proba(self.scaler.transform(X))
        if probs.shape[1] != N_LITH_CLASSES:
            raise AssertionError("eRF probability width must remain 18.")
        return probs

    def predict_frame(self, rows: pd.DataFrame, *, index=None) -> pd.DataFrame:
        return pd.DataFrame(self.predict_proba(rows), columns=GROUP_B_COLUMNS, index=index)


class GroupBPassthrough:
    """Compatibility scaler: Group B probabilities are never re-scaled."""

    def fit(self, values) -> "GroupBPassthrough":
        arr = self._to_array(values)
        self._validate(arr)
        return self

    def transform(self, values) -> np.ndarray:
        arr = self._to_array(values)
        self._validate(arr)
        return arr

    @staticmethod
    def _to_array(values) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            if set(GROUP_B_COLUMNS).issubset(values.columns):
                return values[GROUP_B_COLUMNS].to_numpy(dtype=float)
            return values.to_numpy(dtype=float)
        return np.asarray(values, dtype=float)

    @staticmethod
    def _validate(arr: np.ndarray) -> None:
        if arr.ndim != 2 or arr.shape[1] != N_LITH_CLASSES:
            raise ValueError(f"Group B must have shape (n, {N_LITH_CLASSES}).")
        if not np.isfinite(arr).all() or (arr < -1e-12).any():
            raise ValueError("Group B contains invalid probability values.")
        if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each Group-B probability vector must sum to one.")
