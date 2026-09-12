"""Group C petrophysical proxies from two fixed-parameter XGBoost regressors.

The two outputs are predicted recorded porosity (percentage points) and predicted
log10-permeability (log10 mD).  Only the original five LWD curves are inputs.
Cross-fitting/orchestration is implemented in Round 3; this module deliberately
contains only leakage-safe estimator primitives that fit on rows explicitly
supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.data.io import LWD_COLUMNS
from src.features.group_a import RobustArrayScaler

GROUP_C_COLUMNS = ["predicted_porosity", "predicted_log10_permeability"]

DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


@dataclass
class GroupCPredictions:
    """Two-column predictions aligned to the rows passed to ``predict``."""

    predicted_porosity: np.ndarray
    predicted_log10_permeability: np.ndarray

    def as_array(self) -> np.ndarray:
        return np.column_stack(
            [self.predicted_porosity, self.predicted_log10_permeability]
        ).astype(float)

    def as_frame(self, index=None) -> pd.DataFrame:
        return pd.DataFrame(self.as_array(), columns=GROUP_C_COLUMNS, index=index)


class GroupCRegressors:
    """Pair of XGBoost regressors with manuscript-fixed shared parameters."""

    def __init__(self, *, random_state: int = 20260827, **overrides) -> None:
        params = dict(DEFAULT_XGB_PARAMS)
        params.update(overrides)
        self.params = params
        common = dict(
            **params,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
        self.porosity_model = XGBRegressor(**common)
        self.logk_model = XGBRegressor(**common)
        self.fitted_ = False

    def fit(self, rows: pd.DataFrame) -> "GroupCRegressors":
        required = {*LWD_COLUMNS, "porosity_pct", "permeability_mD"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Group C fit rows missing columns: {sorted(missing)}")
        X = rows[LWD_COLUMNS].to_numpy(dtype=float)
        y_phi = rows["porosity_pct"].to_numpy(dtype=float)
        perm = rows["permeability_mD"].to_numpy(dtype=float)
        if not np.isfinite(X).all() or not np.isfinite(y_phi).all() or not np.isfinite(perm).all():
            raise ValueError("Group C fit rows contain non-finite values.")
        if (perm <= 0).any():
            raise ValueError("permeability_mD must be positive before log10 transform.")
        y_logk = np.log10(perm)
        self.porosity_model.fit(X, y_phi)
        self.logk_model.fit(X, y_logk)
        self.fitted_ = True
        return self

    def predict(self, rows: pd.DataFrame) -> GroupCPredictions:
        if not self.fitted_:
            raise RuntimeError("GroupCRegressors must be fit before predict.")
        X = rows[LWD_COLUMNS].to_numpy(dtype=float)
        return GroupCPredictions(
            predicted_porosity=np.asarray(self.porosity_model.predict(X), dtype=float),
            predicted_log10_permeability=np.asarray(self.logk_model.predict(X), dtype=float),
        )


@dataclass
class GroupCScaler:
    """Robust-scale the two Group-C features separately."""

    robust: RobustArrayScaler | None = None

    @property
    def medians(self) -> np.ndarray | None:
        return None if self.robust is None else self.robust.medians_

    @property
    def iqrs(self) -> np.ndarray | None:
        return None if self.robust is None else self.robust.iqrs_

    def fit(
        self,
        predicted_porosity: np.ndarray,
        predicted_log10_permeability: np.ndarray,
    ) -> "GroupCScaler":
        X = np.column_stack([predicted_porosity, predicted_log10_permeability])
        self.robust = RobustArrayScaler().fit(X)
        return self

    def transform(
        self,
        predicted_porosity: np.ndarray,
        predicted_log10_permeability: np.ndarray,
    ) -> np.ndarray:
        if self.robust is None:
            raise RuntimeError("GroupCScaler must be fit before transform.")
        X = np.column_stack([predicted_porosity, predicted_log10_permeability])
        return self.robust.transform(X)
