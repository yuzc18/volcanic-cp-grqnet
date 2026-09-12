"""Group D local structural context computed on the continuous LWD grid.

The order is deliberately fixed to match manuscript Section 3.3:

1. fit 3σ truncation limits on the classifier-training rows;
2. apply those limits to copies of the continuous five-curve log sequence;
3. compute prev1/next1/mean3/std3(ddof=1)/diff1 within each well;
4. only then match Group-D rows to the core-matched depths;
5. robust-scale the 25 derived features using classifier-training rows only.

No edge padding is performed.  A core sample whose matched grid point lacks the
required neighbours causes an explicit error rather than silent imputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.io import GRID_SPACING_M, LWD_COLUMNS
from src.features.group_a import RawLogClipper, RobustArrayScaler

GROUP_D_BASE_COLUMNS = list(LWD_COLUMNS)
GROUP_D_STATISTICS = ["prev1", "next1", "mean3", "std3", "diff1"]
GROUP_D_COLUMNS = [f"{curve}_{stat}" for curve in GROUP_D_BASE_COLUMNS for stat in GROUP_D_STATISTICS]


def _validate_grid(g: pd.DataFrame, *, spacing_m: float) -> None:
    depth = g["depth_m"].to_numpy(dtype=float)
    if len(depth) > 1 and not np.allclose(np.diff(depth), spacing_m, atol=1e-6, rtol=0):
        raise ValueError(f"Group D requires an uninterrupted {spacing_m:.3f} m grid within each well.")


def build_group_d_continuous(
    continuous_logs: pd.DataFrame,
    *,
    clipper: RawLogClipper,
    causal: bool = False,
    spacing_m: float = GRID_SPACING_M,
) -> pd.DataFrame:
    """Construct 25 Group-D columns on the full continuous sequence.

    Parameters
    ----------
    causal
        ``False`` uses the centered definitions. ``True`` implements the
        strictly causal variant while preserving the same 25 column names:
        the ``next1`` slot stores x(i-2), and mean3/std3 use
        {x(i-2), x(i-1), x(i)}.
    """
    needed = {"log_id", "well_id", "depth_m", *GROUP_D_BASE_COLUMNS}
    missing = needed - set(continuous_logs.columns)
    if missing:
        raise ValueError(f"continuous_logs missing columns for Group D: {sorted(missing)}")

    clipped = clipper.transform(continuous_logs)
    chunks: list[pd.DataFrame] = []
    for well, g0 in clipped.groupby("well_id", sort=False):
        g = g0.sort_values("depth_m", kind="stable").copy()
        _validate_grid(g, spacing_m=spacing_m)
        out = g[["log_id", "well_id", "depth_m"]].copy()

        for curve in GROUP_D_BASE_COLUMNS:
            s = g[curve].astype(float)
            prev1 = s.shift(1)
            diff1 = s - prev1
            if causal:
                replacement = s.shift(2)
                trip = pd.concat([s.shift(2), s.shift(1), s], axis=1)
                next1 = replacement
            else:
                next1 = s.shift(-1)
                trip = pd.concat([s.shift(1), s, s.shift(-1)], axis=1)
            out[f"{curve}_prev1"] = prev1.to_numpy()
            out[f"{curve}_next1"] = next1.to_numpy()
            out[f"{curve}_mean3"] = trip.mean(axis=1, skipna=False).to_numpy()
            out[f"{curve}_std3"] = trip.std(axis=1, ddof=1, skipna=False).to_numpy()
            out[f"{curve}_diff1"] = diff1.to_numpy()
        chunks.append(out)

    return pd.concat(chunks, ignore_index=True)


def match_group_d_to_core(
    core_with_logs: pd.DataFrame,
    group_d_continuous: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the exact Group-D row identified by ``log_id`` to each core sample."""
    if "log_id" not in core_with_logs.columns:
        raise ValueError("core_with_logs must contain log_id from core-to-log registration.")
    cols = ["log_id", *GROUP_D_COLUMNS]
    if not set(cols).issubset(group_d_continuous.columns):
        raise ValueError("group_d_continuous does not contain the required 25 features.")
    merged = core_with_logs[["sample_id", "log_id"]].merge(
        group_d_continuous[cols], on="log_id", how="left", validate="many_to_one"
    )
    if merged[GROUP_D_COLUMNS].isna().any().any():
        bad = merged.loc[merged[GROUP_D_COLUMNS].isna().any(axis=1), "sample_id"].tolist()[:10]
        raise ValueError(
            "At least one core-matched row lacks the required Group-D neighbours; "
            f"no padding is allowed. Examples: {bad}"
        )
    return merged[["sample_id", *GROUP_D_COLUMNS]]


@dataclass
class GroupDScaler:
    """Robust median/IQR scaler for already-derived Group-D features."""

    columns: list[str] = field(default_factory=lambda: list(GROUP_D_COLUMNS))
    robust: RobustArrayScaler | None = None

    @property
    def medians(self) -> np.ndarray | None:
        return None if self.robust is None else self.robust.medians_

    @property
    def iqrs(self) -> np.ndarray | None:
        return None if self.robust is None else self.robust.iqrs_

    def fit(self, df: pd.DataFrame) -> "GroupDScaler":
        X = df[self.columns].to_numpy(dtype=float)
        self.robust = RobustArrayScaler().fit(X)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.robust is None:
            raise RuntimeError("GroupDScaler must be fit before transform.")
        return self.robust.transform(df[self.columns].to_numpy(dtype=float))


# Legacy name retained for imports; this function now requires data that are
# already continuous and a fitted clipper rather than silently padding edges.
def build_group_d_columns(
    df: pd.DataFrame,
    *,
    clipper: RawLogClipper | None = None,
    causal: bool = False,
) -> pd.DataFrame:
    if "well" in df.columns and "well_id" not in df.columns:
        df = df.rename(columns={"well": "well_id"}).copy()
    if "log_id" not in df.columns:
        tmp = df.copy()
        tmp["log_id"] = [f"ROW-{i}" for i in range(len(tmp))]
    else:
        tmp = df.copy()
    clipper = clipper or RawLogClipper().fit(tmp)
    result = build_group_d_continuous(tmp, clipper=clipper, causal=causal)
    result.index = tmp.sort_values(["well_id", "depth_m"], kind="stable").index
    return result[GROUP_D_COLUMNS].sort_index()
