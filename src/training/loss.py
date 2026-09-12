"""Training loss: class-weighted cross-entropy with label smoothing.

Implements the loss specified in Section 3.6 of the paper:

    L = - sum_c  w_c  y_smoothed[c]  log softmax(logits)[c]

where:
    - w_c          = class weights, inversely proportional to class
                     frequency in the training set;
    - y_smoothed   = (1 - epsilon) * one_hot(y) + epsilon / K
                     for K = 3 classes and epsilon = 0.05 (Mueller et al., 2019).

The CE is computed via ``log_softmax`` for numerical stability — the
model returns raw logits (see :mod:`src.models.grq_net`), so no explicit
softmax is needed in the forward path.

Notation matches Section 3.6:

    - "epsilon = 0.05" label smoothing (paper)
    - class weights computed from training-set class frequencies
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------
def compute_inverse_frequency_weights(
    labels: np.ndarray, n_classes: int = 3
) -> torch.Tensor:
    """Compute class weights as ``N / (K * n_c)``.

    This is the canonical inverse-frequency normalization that makes the
    average weight equal to 1 regardless of class imbalance and matches
    the standard interpretation of "class weights computed as the inverse
    of the class frequencies in the training set" (Section 3.6).

    Parameters
    ----------
    labels
        1-D array of integer class labels for the training rows ONLY.
    n_classes
        Number of classes (paper: 3).

    Returns
    -------
    torch.Tensor
        Shape ``(n_classes,)``, dtype ``float32``. Index ``c`` holds the
        weight for class ``c``. If a class is absent from ``labels`` (a
        degenerate case that should not happen in our well-level folds),
        its weight is set to 1.0 rather than ``inf``.
    """
    counts = Counter(labels.tolist())
    total = float(len(labels))
    weights = np.empty(n_classes, dtype=np.float32)
    for c in range(n_classes):
        n_c = counts.get(c, 0)
        weights[c] = total / (n_classes * n_c) if n_c > 0 else 1.0
    return torch.from_numpy(weights)


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------
class SmoothedWeightedCrossEntropy(nn.Module):
    """Class-weighted cross-entropy with epsilon-smoothed labels.

    Parameters
    ----------
    class_weights
        Shape ``(n_classes,)``. Typically produced by
        :func:`compute_inverse_frequency_weights`.
    label_smoothing
        Epsilon in ``[0, 1)``. Paper value: 0.05.
    n_classes
        Number of classes (paper: 3).

    Notes
    -----
    The label-smoothing convention used here is the standard one from
    Mueller et al. (2019):

        y_smoothed[true_class] = 1 - epsilon + epsilon / K
        y_smoothed[other]      = epsilon / K

    which makes the smoothed vector sum to exactly 1 regardless of
    epsilon. Class weighting is applied per-class then averaged across
    the batch (mean reduction), matching ``torch.nn.CrossEntropyLoss``'s
    default behavior.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        *,
        label_smoothing: float = 0.05,
        n_classes: int = 3,
    ):
        super().__init__()
        if class_weights.shape != (n_classes,):
            raise ValueError(
                f"class_weights must have shape ({n_classes},), "
                f"got {tuple(class_weights.shape)}"
            )
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1).")

        # Register as a non-trainable buffer so it moves with .to(device)
        self.register_buffer("class_weights", class_weights.float())
        self.label_smoothing = float(label_smoothing)
        self.n_classes = int(n_classes)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the scalar loss.

        Parameters
        ----------
        logits
            Shape ``(batch, n_classes)``: raw model outputs.
        targets
            Shape ``(batch,)``: integer class labels.
        """
        if logits.dim() != 2 or logits.size(1) != self.n_classes:
            raise ValueError(
                f"logits shape must be (batch, {self.n_classes}); "
                f"got {tuple(logits.shape)}"
            )
        if targets.dim() != 1 or targets.size(0) != logits.size(0):
            raise ValueError("targets must be 1-D and aligned with logits.")

        # ---- Build smoothed targets ---------------------------------------
        # off-value for non-true classes; on-value for the true class
        K = self.n_classes
        eps = self.label_smoothing
        on_value = 1.0 - eps + eps / K
        off_value = eps / K

        with torch.no_grad():
            y_smoothed = torch.full(
                (logits.size(0), K),
                off_value,
                dtype=logits.dtype,
                device=logits.device,
            )
            y_smoothed.scatter_(1, targets.unsqueeze(1), on_value)

        # ---- Weighted log-softmax cross entropy ---------------------------
        log_probs = F.log_softmax(logits, dim=1)                    # (B, K)
        per_class_loss = -y_smoothed * log_probs                    # (B, K)

        # Apply class weights: multiply each column by the corresponding weight
        weighted_per_class = per_class_loss * self.class_weights.unsqueeze(0)
        # Sum over classes, average over the batch (standard 'mean' reduction)
        per_sample_loss = weighted_per_class.sum(dim=1)             # (B,)
        return per_sample_loss.mean()
