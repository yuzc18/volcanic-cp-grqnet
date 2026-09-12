"""Assembly and scaling of the 50-dimensional unified feature table (UFT)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.io import LWD_COLUMNS
from src.features.group_a import RawLogClipper, RobustArrayScaler
from src.features.group_b import GROUP_B_COLUMNS, GroupBPassthrough
from src.features.group_c import GROUP_C_COLUMNS, GroupCPredictions
from src.features.group_d import GROUP_D_COLUMNS


@dataclass(frozen=True)
class UFTLayout:
    dim_a: int = 5
    dim_b: int = 18
    dim_c: int = 2
    dim_d: int = 25

    @property
    def total(self) -> int:
        return self.dim_a + self.dim_b + self.dim_c + self.dim_d

    @property
    def group_dims(self) -> tuple[int, int, int, int]:
        return self.dim_a, self.dim_b, self.dim_c, self.dim_d

    @property
    def slice_a(self) -> slice:
        return slice(0, 5)

    @property
    def slice_b(self) -> slice:
        return slice(5, 23)

    @property
    def slice_c(self) -> slice:
        return slice(23, 25)

    @property
    def slice_d(self) -> slice:
        return slice(25, 50)


UFT_LAYOUT = UFTLayout()
UFT_COLUMNS = list(LWD_COLUMNS) + GROUP_B_COLUMNS + GROUP_C_COLUMNS + GROUP_D_COLUMNS


def _group_c_array(values) -> np.ndarray:
    if isinstance(values, GroupCPredictions):
        return values.as_array()
    if isinstance(values, pd.DataFrame):
        return values[GROUP_C_COLUMNS].to_numpy(dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Group C must have shape (n, 2).")
    return arr


def _group_b_array(values) -> np.ndarray:
    return GroupBPassthrough().fit(values).transform(values)


def _group_d_array(values) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        return values[GROUP_D_COLUMNS].to_numpy(dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 25:
        raise ValueError("Group D must have shape (n, 25).")
    return arr


@dataclass
class UFTPreprocessor:
    """Fit A/C/D robust scalers on classifier-training rows; keep B unscaled.

    The same ``RawLogClipper`` used to derive Group D should be supplied here so
    Group A and D share the training-only 3σ limits applied to copies of the raw
    curves.  Group B remains as raw probabilities.  Group C's two proxy columns
    are scaled separately by the generic per-column median/IQR transform.
    """

    raw_clipper: RawLogClipper | None = None
    scaler_a: RobustArrayScaler | None = None
    scaler_c: RobustArrayScaler | None = None
    scaler_d: RobustArrayScaler | None = None
    fitted: bool = False

    def fit_on(
        self,
        train_core_with_logs: pd.DataFrame,
        train_group_b,
        train_group_c,
        train_group_d,
        *,
        raw_clipper: RawLogClipper | None = None,
    ) -> "UFTPreprocessor":
        n = len(train_core_with_logs)
        B = _group_b_array(train_group_b)
        C = _group_c_array(train_group_c)
        D = _group_d_array(train_group_d)
        if not (len(B) == len(C) == len(D) == n):
            raise ValueError("All UFT groups must contain the same number of rows.")

        self.raw_clipper = raw_clipper or RawLogClipper().fit(train_core_with_logs)
        A_clipped = self.raw_clipper.transform_array(
            train_core_with_logs[LWD_COLUMNS].to_numpy(dtype=float)
        )
        self.scaler_a = RobustArrayScaler().fit(A_clipped)
        self.scaler_c = RobustArrayScaler().fit(C)
        self.scaler_d = RobustArrayScaler().fit(D)
        self.fitted = True
        return self

    def transform(
        self,
        core_with_logs: pd.DataFrame,
        group_b,
        group_c,
        group_d,
    ) -> np.ndarray:
        if not self.fitted or self.raw_clipper is None:
            raise RuntimeError("UFTPreprocessor must be fit before transform.")
        assert self.scaler_a is not None and self.scaler_c is not None and self.scaler_d is not None
        n = len(core_with_logs)
        B = _group_b_array(group_b)
        C = _group_c_array(group_c)
        D = _group_d_array(group_d)
        if not (len(B) == len(C) == len(D) == n):
            raise ValueError("All UFT groups must contain the same number of rows.")

        A_raw = core_with_logs[LWD_COLUMNS].to_numpy(dtype=float)
        A = self.scaler_a.transform(self.raw_clipper.transform_array(A_raw))
        C_scaled = self.scaler_c.transform(C)
        D_scaled = self.scaler_d.transform(D)
        X = np.concatenate([A, B, C_scaled, D_scaled], axis=1)
        if X.shape != (n, 50):
            raise AssertionError(f"UFT must be (n, 50), got {X.shape}.")
        return X

    def transform_frame(
        self,
        core_with_logs: pd.DataFrame,
        group_b,
        group_c,
        group_d,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.transform(core_with_logs, group_b, group_c, group_d),
            columns=UFT_COLUMNS,
            index=core_with_logs.index,
        )


def assemble_unscaled_uft(
    core_with_logs: pd.DataFrame,
    group_b,
    group_c,
    group_d,
) -> pd.DataFrame:
    """Create a named 50-column table before A/C/D robust scaling.

    This is mainly useful for auditing feature order.  Group A here contains
    raw current-depth values; study modeling should use :class:`UFTPreprocessor`
    so Group A is 3σ-clipped and A/C/D are robust-scaled.
    """
    B = _group_b_array(group_b)
    C = _group_c_array(group_c)
    D = _group_d_array(group_d)
    A = core_with_logs[LWD_COLUMNS].to_numpy(dtype=float)
    X = np.concatenate([A, B, C, D], axis=1)
    if X.shape[1] != 50:
        raise AssertionError("UFT width changed unexpectedly.")
    return pd.DataFrame(X, columns=UFT_COLUMNS, index=core_with_logs.index)
