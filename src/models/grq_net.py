"""GRQ-Net: Group-aware Residual Fusion Network (Section 3.4).

This module implements the architecture defined by Equations (4) through
(10) of the paper, and Table 3 (hyperparameter configuration).

Architecture overview
---------------------
Input: 50-dim UFT vector ``x = [x_A; x_B; x_C; x_D]`` with
``(d_A, d_B, d_C, d_D) = (5, 18, 2, 25)``.

    (i)   Intra-group independent encoding (Eq. 4)
              e_k = LayerNorm( GELU( W_k x_k ) ),  k in {A, B, C, D}
              All four embeddings share the same dimensionality d_g = 48.

    (ii)  Full-context group reweighting (Eq. 5-6)
              context = concat(e_A, e_B, e_C, e_D)        # in R^{4 d_g}
              alpha   = Sigmoid( W2 ReLU( W1 context ) )  # in (0,1)^4
              fused   = concat(alpha_A * e_A, alpha_B * e_B,
                               alpha_C * e_C, alpha_D * e_D)
          Sigmoid (NOT Softmax): the four groups are complementary, not
          competitive — the paper argues this explicitly.

    (iii) Gated residual learning (Eq. 7-9), repeated L = 4 times
              main(x)  = ReLU( W_m x )
              gate(x)  = Sigmoid( W_g x )
              y        = LayerNorm( x + Dropout(main(x) * gate(x)) )

    (iv)  Classification head (Eq. 10)
              logits = W_2 GELU( W_1 fused )    # 2-layer FC
              P      = Softmax(logits) over 3 classes

The group weights alpha are exported as an auxiliary output to support
the interpretability analysis in Section 5.1 (Fig. 10).

Parameter count target: 319,815 (~320K, per Table 3). The
``count_parameters`` method asserts this exactly when the default
hyperparameters from ``configs/default.yaml`` are used.

Numerical notes
---------------
* Softmax is applied OUTSIDE the model in inference paths; the model
  returns raw logits so the training loss can use ``log_softmax`` for
  numerical stability and CP can take softmax probabilities directly.
* The model runs on CPU efficiently; the paper reports 0.49 ms per
  sample on CPU. We do not assume CUDA anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration dataclass (mirrors configs/default.yaml::grqnet)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GRQNetConfig:
    """Static hyperparameter configuration for GRQ-Net (Table 3 of the paper)."""

    group_dims: tuple[int, ...] = (5, 18, 2, 25)  # active semantic-group dimensions
    d_group: int = 48                                       # d_g (Table 3)
    attention_hidden: int = 32                              # h in Eq. (5)
    n_residual_blocks: int = 4                              # L = 4 (Table 3)
    n_classes: int = 3                                      # High / Medium / Low
    classification_head_hidden: int = 64                    # hidden dim of Eq. (10)
    dropout: float = 0.12                                   # see configs/default.yaml

    @property
    def d_model(self) -> int:
        """Fused-representation dimensionality, ``d_model = 4 * d_g = 192``."""
        return self.d_group * len(self.group_dims)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------
class GroupEncoder(nn.Module):
    """Eq. (4): intra-group independent encoder.

        e_k = LayerNorm( GELU( W_k x_k ) )

    Table 3(a) lists the group encoders among the three dropout
    locations, so the Dropout applied here is specified by the
    manuscript rather than added by this implementation. Dropout has no
    trainable parameters and therefore does not affect the reported
    319,815-parameter count.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.activation(self.linear(x))))


class FullContextGroupAttention(nn.Module):
    """Eq. (5)-(6): full-context group reweighting.

        alpha = Sigmoid( W2 ReLU( W1 [e_A; e_B; e_C; e_D] ) )

    Returns the four sample-level group weights and the rescaled, fused
    representation.

    Important
    ---------
    The activation between W1 and W2 is ReLU (a lightweight 2-layer MLP
    head). The OUTPUT activation is Sigmoid, NOT Softmax: the paper
    argues groups are complementary, so each alpha_k must be free to be
    high independently of the others.
    """

    def __init__(self, n_groups: int, d_group: int, hidden: int):
        super().__init__()
        self.n_groups = n_groups
        self.d_group = d_group
        d_context = n_groups * d_group
        self.fc1 = nn.Linear(d_context, hidden)
        self.fc2 = nn.Linear(hidden, n_groups)

    def forward(
        self, embeddings: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        embeddings
            List of 4 tensors, each shape ``(batch, d_group)``.

        Returns
        -------
        fused : torch.Tensor
            Shape ``(batch, n_groups * d_group)`` — the rescaled,
            concatenated group embeddings ready for the residual stack.
        alpha : torch.Tensor
            Shape ``(batch, n_groups)`` — the sample-level group weights,
            useful as an interpretability auxiliary output.
        """
        # Stack as (batch, n_groups, d_group) then flatten to (batch, n_groups*d_group)
        stacked = torch.stack(embeddings, dim=1)
        context = stacked.reshape(stacked.size(0), -1)

        # alpha = Sigmoid( fc2( ReLU( fc1( context ) ) ) ) in (0,1)^{n_groups}
        alpha = torch.sigmoid(self.fc2(F.relu(self.fc1(context))))

        # Scale each group embedding by its alpha and re-flatten
        scaled = stacked * alpha.unsqueeze(2)         # (batch, n_groups, d_group)
        fused = scaled.reshape(stacked.size(0), -1)   # (batch, n_groups*d_group)
        return fused, alpha


class GatedResidualBlock(nn.Module):
    """Eq. (7)-(9): gated residual block with GLU-style gating.

        main(x) = ReLU( W_m x )
        gate(x) = Sigmoid( W_g x )
        y       = LayerNorm( x + Dropout(main(x) * gate(x)) )

    The ReLU on the main branch is required by manuscript Eq. (7).
    Dropout is applied to the gated path before the residual addition,
    matching Eq. (9). ReLU adds no trainable parameters, so the reported
    319,815-parameter count is unchanged.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.main = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.relu(self.main(x)) * torch.sigmoid(self.gate(x))
        return self.norm(x + self.dropout(gated))


class ClassificationHead(nn.Module):
    """Eq. (10): 2-layer fully connected classifier.

        logits = W_2 GELU( W_1 fused )

    Returns RAW logits (no Softmax) for numerical stability of the
    log-softmax-based cross-entropy loss. The downstream code applies
    Softmax explicitly when probabilities are needed (e.g., CP).
    """

    def __init__(self, d_model: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


# ---------------------------------------------------------------------------
# GRQ-Net main model
# ---------------------------------------------------------------------------
class GRQNet(nn.Module):
    """Group-aware Residual Fusion Network (Section 3.4 of the paper).

    Wires together the four sub-modules above into the architecture
    illustrated in Figure 4 of the paper.

    Parameters
    ----------
    config
        Static hyperparameter configuration. Defaults match Table 3.
    """

    def __init__(self, config: GRQNetConfig | None = None):
        super().__init__()
        self.config = config if config is not None else GRQNetConfig()

        # ---- (i) Four intra-group encoders ---------------------------------
        self.encoders = nn.ModuleList([
            GroupEncoder(in_dim=d_in, out_dim=self.config.d_group,
                         dropout=self.config.dropout)
            for d_in in self.config.group_dims
        ])

        # ---- (ii) Full-context group reweighting ---------------------------
        self.group_attention = FullContextGroupAttention(
            n_groups=len(self.config.group_dims),
            d_group=self.config.d_group,
            hidden=self.config.attention_hidden,
        )

        # ---- (iii) L gated residual blocks ---------------------------------
        self.residual_blocks = nn.ModuleList([
            GatedResidualBlock(d_model=self.config.d_model,
                               dropout=self.config.dropout)
            for _ in range(self.config.n_residual_blocks)
        ])

        # ---- (iv) Classification head --------------------------------------
        self.head = ClassificationHead(
            d_model=self.config.d_model,
            hidden=self.config.classification_head_hidden,
            n_classes=self.config.n_classes,
            dropout=self.config.dropout,
        )

        # ---- Group-slice positions for forward pass -----------------------
        # Precompute cumulative offsets so forward() can slice efficiently.
        cum = [0]
        for d in self.config.group_dims:
            cum.append(cum[-1] + d)
        self._group_starts: tuple[int, ...] = tuple(cum[:-1])
        self._group_ends: tuple[int, ...] = tuple(cum[1:])
        assert cum[-1] == sum(self.config.group_dims)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, *, return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute class logits (and optionally group weights).

        Parameters
        ----------
        x
            Shape ``(batch, sum(config.group_dims))``. The default manuscript model uses 50 features.
        return_attention
            If True, also return the per-sample group weights ``alpha``
            (shape ``(batch, 4)``). Used by interpretability analyses
            (Section 5.1, Figure 10b-c).

        Returns
        -------
        logits : torch.Tensor
            Shape ``(batch, 3)``. Apply ``softmax`` outside the model
            to obtain class probabilities.
        alpha : torch.Tensor, optional
            Group weights, only returned if ``return_attention=True``.
        """
        # ---- (i) Per-group independent encoding ---------------------------
        embeddings: list[torch.Tensor] = []
        for enc, lo, hi in zip(
            self.encoders, self._group_starts, self._group_ends, strict=True
        ):
            embeddings.append(enc(x[:, lo:hi]))

        # ---- (ii) Full-context reweighting --------------------------------
        fused, alpha = self.group_attention(embeddings)

        # ---- (iii) Gated residual stack -----------------------------------
        h = fused
        for block in self.residual_blocks:
            h = block(h)

        # ---- (iv) Classification head -------------------------------------
        logits = self.head(h)

        if return_attention:
            return logits, alpha
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        """Per-component parameter count, for documentation and debugging."""
        return {
            "encoders": sum(p.numel() for p in self.encoders.parameters()),
            "group_attention": sum(p.numel() for p in self.group_attention.parameters()),
            "residual_blocks": sum(p.numel() for p in self.residual_blocks.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
            "total": self.count_parameters(),
        }
