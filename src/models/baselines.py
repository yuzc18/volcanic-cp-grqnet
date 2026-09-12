"""Baseline neural architectures reported in Table 3(a).

The manuscript specifies the principal architecture choices and parameter counts.
MLP, LSTM, and Transformer are implemented directly in PyTorch so they can use
exactly the same optimizer/loss/scheduler/early-stopping loop as GRQ-Net.
TabNet is wrapped lazily from ``pytorch-tabnet==4.1.0``.

Unreported low-level defaults are kept explicit in ``configs/baselines.yaml``;
they are never presented as manuscript facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


EXPECTED_PARAMETER_COUNTS = {
    "mlp": 25_315,
    "lstm": 50_627,
    "transformer": 153_475,
    "tabnet": 108_728,
}


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int = 50, hidden_dims=(128, 96, 64), dropout: float = 0.3, n_classes: int = 3):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d0, d1 in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(d0, d1), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMBaseline(nn.Module):
    """Each UFT row is a length-50 sequence of scalar tokens."""

    def __init__(self, hidden_size: int = 64, n_layers: int = 2, dropout: float = 0.3, n_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)  # (batch, 50, 1)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1, :])


class TransformerBaseline(nn.Module):
    """Scalar-token Transformer described in Section 3.6."""

    def __init__(
        self,
        n_tokens: int = 50,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.scalar_projection = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.n_tokens:
            raise ValueError(f"Transformer expects {self.n_tokens} scalar tokens, got {x.shape[1]}.")
        h = self.scalar_projection(x.unsqueeze(-1)) + self.position
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


class TabNetBaseline(nn.Module):
    """Thin nn.Module wrapper around pytorch-tabnet's TabNet network."""

    def __init__(
        self,
        input_dim: int = 50,
        n_classes: int = 3,
        n_steps: int = 4,
        n_a: int = 32,
        n_d: int = 32,
        gamma: float = 1.5,
        **library_defaults: Any,
    ):
        super().__init__()
        try:
            from pytorch_tabnet.tab_network import TabNet
        except ImportError as exc:  # pragma: no cover - depends on optional runtime install
            raise ImportError(
                "TabNet baseline requires pytorch-tabnet==4.1.0; install requirements.txt."
            ) from exc
        # pytorch-tabnet 4.1.0 defaults ``group_attention_matrix`` to an empty
        # list, which ``EmbeddingGenerator`` then calls ``.to()`` on.  Its
        # ``TabNetClassifier`` wrapper builds the matrix before instantiating the
        # network, so constructing ``TabNet`` directly has to supply it.  The
        # identity matrix is the no-grouping case (one attention group per
        # feature) and reproduces the Table 3(a) parameter count of 108,728.
        library_defaults.setdefault("group_attention_matrix", torch.eye(input_dim))
        self.tabnet = TabNet(
            input_dim=input_dim,
            output_dim=n_classes,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            **library_defaults,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tabnet(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass(frozen=True)
class NeuralBaselineBuild:
    name: str
    model: nn.Module
    expected_parameters: int
    actual_parameters: int


def build_neural_baseline(name: str, cfg: dict, *, input_dim: int = 50, n_classes: int = 3) -> NeuralBaselineBuild:
    name = name.lower()
    b = cfg["neural_baselines"][name]
    if name == "mlp":
        model = MLPBaseline(input_dim, tuple(b["hidden_dims"]), float(b["dropout"]), n_classes)
    elif name == "lstm":
        model = LSTMBaseline(int(b["hidden_size"]), int(b["n_layers"]), float(b["dropout"]), n_classes)
    elif name == "transformer":
        model = TransformerBaseline(
            n_tokens=input_dim,
            d_model=int(b["d_model"]),
            n_heads=int(b["n_heads"]),
            n_layers=int(b["n_encoder_layers"]),
            d_ff=int(b["d_ff"]),
            dropout=float(b.get("dropout", 0.1)),
            n_classes=n_classes,
        )
    elif name == "tabnet":
        extras = dict(b.get("library_defaults", {}))
        model = TabNetBaseline(
            input_dim=input_dim,
            n_classes=n_classes,
            n_steps=int(b["n_steps"]),
            n_a=int(b["n_a"]),
            n_d=int(b["n_d"]),
            gamma=float(b["relaxation"]),
            **extras,
        )
    else:
        raise KeyError(f"Unknown neural baseline: {name}")
    actual = count_trainable_parameters(model)
    expected = int(b["trainable_parameters"])
    return NeuralBaselineBuild(name, model, expected, actual)
