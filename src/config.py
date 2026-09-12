"""Configuration loading and reproducibility utilities.

Centralizes:
    1. Loading and validating YAML configuration files.
    2. Setting global random seeds across numpy / random / torch / cuda.
    3. Enabling deterministic algorithms in PyTorch.

Paper-facing configuration is split into topic-specific YAML files under ``configs/``.
``configs/default.yaml`` is retained only as a backward-compatible aggregate for
pre-revision modules that have not yet been migrated.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
WELLS_CONFIG_PATH = CONFIG_DIR / "wells.yaml"
PAPER_CONFIG_FILES = [
    "seeds.yaml",
    "data.yaml",
    "labels.yaml",
    "features.yaml",
    "erf.yaml",
    "upstream_xgb.yaml",
    "grqnet.yaml",
    "baselines.yaml",
    "search_spaces.yaml",
    "conformal.yaml",
    "statistics.yaml",
    "benchmark.yaml",
]


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass
class WellEntry:
    """Single well record from ``wells.yaml``."""

    name: str
    n_samples: int
    depth_min: float
    depth_max: float
    dominant_lithology: str
    perm_min_mD: float
    perm_max_mD: float


@dataclass
class WellsConfig:
    """Parsed contents of ``wells.yaml``."""

    training_wells: list[WellEntry] = field(default_factory=list)
    blind_wells: list[WellEntry] = field(default_factory=list)
    totals: dict[str, int] = field(default_factory=dict)

    @property
    def training_well_names(self) -> list[str]:
        return [w.name for w in self.training_wells]

    @property
    def blind_well_names(self) -> list[str]:
        return [w.name for w in self.blind_wells]

    @property
    def all_well_names(self) -> list[str]:
        return self.training_well_names + self.blind_well_names


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the backward-compatible aggregate configuration."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paper_configs(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load all topic-specific paper-facing YAML files.

    The return value maps each filename stem (for example ``"labels"``) to
    its parsed YAML content. Keeping topics separate makes it harder for a
    change in one experimental component to silently alter another.
    """
    config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    out: dict[str, Any] = {}
    for filename in PAPER_CONFIG_FILES:
        path = config_dir / filename
        with open(path, encoding="utf-8") as f:
            out[path.stem] = yaml.safe_load(f)
    return out


def load_wells_config(path: str | Path | None = None) -> WellsConfig:
    """Load and validate the well roster.

    Raises
    ------
    AssertionError
        If sample counts in the YAML disagree with the totals block.
    """
    path = Path(path) if path is not None else WELLS_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    training = [WellEntry(**w) for w in raw["training_wells"]]
    blind = [WellEntry(**w) for w in raw["blind_wells"]]
    totals = raw["totals"]

    # ---- Sanity check: counts must add up to the published totals ---------
    sum_train = sum(w.n_samples for w in training)
    sum_blind = sum(w.n_samples for w in blind)
    assert sum_train == totals["training_samples"], (
        f"Training well samples sum to {sum_train}, expected "
        f"{totals['training_samples']} (Table 1)."
    )
    assert sum_blind == totals["blind_samples"], (
        f"Blind well samples sum to {sum_blind}, expected "
        f"{totals['blind_samples']} (Table 1)."
    )
    assert sum_train + sum_blind == totals["total_samples"], (
        "Training + blind sample counts do not equal total_samples."
    )
    return WellsConfig(training_wells=training, blind_wells=blind, totals=totals)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Seed every PRNG that the pipeline relies on.

    Parameters
    ----------
    seed
        Master seed used for ``random``, ``numpy``, and ``torch`` (CPU + CUDA).

    Notes
    -----
    Per-component seeds (XGBoost OOF, per-fold CV, CP split, bootstrap) are
    stored under ``configs/`` so that components can be re-run independently
    with consistent results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_deterministic_mode(cfg_determinism: dict[str, bool]) -> None:
    """Force PyTorch into a deterministic regime.

    Parameters
    ----------
    cfg_determinism
        Determinism sub-dictionary from the repository configuration.

    Notes
    -----
    Setting ``CUBLAS_WORKSPACE_CONFIG`` is required on recent CUDA versions
    for ``torch.use_deterministic_algorithms(True)`` to actually take effect
    on CUDA operations.
    """
    if cfg_determinism.get("torch_deterministic", True):
        # Workspace config must be set before any CUDA op runs.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cudnn.benchmark = cfg_determinism.get("cudnn_benchmark", False)
    torch.backends.cudnn.deterministic = cfg_determinism.get("cudnn_deterministic", True)


def setup_reproducibility(cfg: dict[str, Any]) -> None:
    """One-stop call: seed everything + enable determinism."""
    set_global_seed(cfg["seed"]["global"])
    enable_deterministic_mode(cfg["determinism"])
