"""Table-5 feature-group ablation definitions and helpers.

For GRQ-Net ablations, omitted semantic groups are removed from the model input
and the network is re-instantiated with encoders only for the active groups.
This is the paper-facing implementation used by ``scripts/run_ablation.py``.
All other architecture and training settings remain unchanged.
"""
from __future__ import annotations

import numpy as np

from src.models.grq_net import GRQNetConfig

GROUP_SLICES = {"A": slice(0, 5), "B": slice(5, 23), "C": slice(23, 25), "D": slice(25, 50)}
GROUP_DIMS = {"A": 5, "B": 18, "C": 2, "D": 25}
TABLE5_CONFIGS = {
    "A": ("A",),
    "A+B": ("A", "B"),
    "A+B+C": ("A", "B", "C"),
    "A+B+D": ("A", "B", "D"),
    "A+C+D": ("A", "C", "D"),
    "A+B+C+D": ("A", "B", "C", "D"),
}


def select_groups(X: np.ndarray, groups: tuple[str, ...]) -> np.ndarray:
    """Concatenate only the active UFT groups, in A-B-C-D order."""
    return np.concatenate([X[:, GROUP_SLICES[g]] for g in groups], axis=1)


def grq_ablation_inputs_and_config(X: np.ndarray, groups: tuple[str, ...]):
    """Return active-group input and the matching re-instantiated GRQ-Net config."""
    Xin = select_groups(X, groups)
    cfg = GRQNetConfig(group_dims=tuple(GROUP_DIMS[g] for g in groups))
    return Xin, cfg
