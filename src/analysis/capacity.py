"""Seven one-factor-at-a-time GRQ-Net capacity configurations."""
from __future__ import annotations

from dataclasses import replace

from src.models.grq_net import GRQNetConfig


def capacity_scan_configs() -> list[tuple[str, GRQNetConfig]]:
    base = GRQNetConfig()
    return [
        ("reported", base),
        ("L=2", replace(base, n_residual_blocks=2)),
        ("L=6", replace(base, n_residual_blocks=6)),
        ("d_g=32", replace(base, d_group=32)),
        ("d_g=64", replace(base, d_group=64)),
        ("dropout=0", replace(base, dropout=0.0)),
        ("dropout=0.30", replace(base, dropout=0.30)),
    ]
