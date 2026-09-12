"""Group A preprocessing: 3σ-clipped raw LWD responses + robust scaling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.io import LWD_COLUMNS

GROUP_A_COLUMNS = list(LWD_COLUMNS)


@dataclass
class RawLogClipper:
    """Per-curve 3σ truncation fitted on classifier-training rows only."""

    columns: list[str] = field(default_factory=lambda: list(GROUP_A_COLUMNS))
    sigma: float = 3.0
    means_: np.ndarray | None = None
    stds_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "RawLogClipper":
        X = df[self.columns].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("RawLogClipper fit data contain non-finite values.")
        self.means_ = X.mean(axis=0)
        self.stds_ = X.std(axis=0, ddof=0)
        self.stds_ = np.where(self.stds_ > 0, self.stds_, 1.0)
        return self

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("RawLogClipper must be fit before transform.")
        X = np.asarray(X, dtype=float)
        lo = self.means_ - self.sigma * self.stds_
        hi = self.means_ + self.sigma * self.stds_
        return np.clip(X, lo, hi)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.columns] = self.transform_array(out[self.columns].to_numpy(dtype=float))
        return out


@dataclass
class RobustArrayScaler:
    """Median/IQR scaler with deterministic handling of zero-IQR columns."""

    medians_: np.ndarray | None = None
    iqrs_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RobustArrayScaler":
        X = np.asarray(X, dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("Robust scaler fit data contain non-finite values.")
        self.medians_ = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.iqrs_ = np.where(iqr > 0, iqr, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.medians_ is None or self.iqrs_ is None:
            raise RuntimeError("RobustArrayScaler must be fit before transform.")
        X = np.asarray(X, dtype=float)
        return (X - self.medians_) / self.iqrs_


@dataclass
class GroupAScaler:
    """Fit the shared raw-log clipper and Group-A robust scaler."""

    columns: list[str] = field(default_factory=lambda: list(GROUP_A_COLUMNS))
    clipper: RawLogClipper | None = None
    robust: RobustArrayScaler | None = None

    # Backward-compatible public attributes used by legacy tests.
    @property
    def means(self) -> np.ndarray | None:
        return None if self.clipper is None else self.clipper.means_

    @property
    def stds(self) -> np.ndarray | None:
        return None if self.clipper is None else self.clipper.stds_

    @property
    def medians(self) -> np.ndarray | None:
        return None if self.robust is None else self.robust.medians_

    @property
    def iqrs(self) -> np.ndarray | None:
        return None if self.robust is None else self.robust.iqrs_

    def fit(self, df: pd.DataFrame) -> "GroupAScaler":
        self.clipper = RawLogClipper(columns=list(self.columns)).fit(df)
        clipped = self.clipper.transform_array(df[self.columns].to_numpy(dtype=float))
        self.robust = RobustArrayScaler().fit(clipped)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.clipper is None or self.robust is None:
            raise RuntimeError("GroupAScaler must be fit before transform.")
        clipped = self.clipper.transform_array(df[self.columns].to_numpy(dtype=float))
        return self.robust.transform(clipped)
