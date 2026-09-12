"""Single-fold GRQ-Net training loop (Section 3.6, Table 3).

Implements the shared training protocol:
    - Optimizer  : AdamW (lr = 8e-4, weight_decay = 5e-3)
    - Scheduler  : CosineAnnealingWarmRestarts (T_0 = 30, T_mult = 2)
    - Grad clip  : max norm 1.0
    - Batch size : 48
    - Early stop : monitor val macro-F1, patience 30, keep best checkpoint
    - Loss       : class-weighted CE + label smoothing (eps = 0.05)

The trainer is deliberately stateless across folds: each call to
:func:`train_one_fold` constructs a fresh model, optimizer, and scheduler
so that no information bleeds between folds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.grq_net import GRQNet, GRQNetConfig
from src.training.loss import (
    SmoothedWeightedCrossEntropy,
    compute_inverse_frequency_weights,
)
from src.training.metrics import compute_metrics


@dataclass
class TrainConfig:
    """Training hyperparameters (Table 3)."""

    lr: float = 8e-4
    weight_decay: float = 5e-3
    label_smoothing: float = 0.05
    batch_size: int = 48
    max_epochs: int = 200
    early_stop_patience: int = 30
    grad_clip_max_norm: float = 1.0
    cosine_T_0: int = 30
    cosine_T_mult: int = 2
    n_classes: int = 3

    @classmethod
    def from_yaml(cls, cfg: dict) -> "TrainConfig":
        """Build from the ``training`` block of ``configs/default.yaml``."""
        t = cfg["training"]
        return cls(
            lr=float(t["optimizer"]["lr"]),
            weight_decay=float(t["optimizer"]["weight_decay"]),
            label_smoothing=float(t["loss"]["label_smoothing"]),
            batch_size=int(t["batch_size"]),
            max_epochs=int(t["max_epochs"]),
            early_stop_patience=int(t["early_stop"]["patience"]),
            grad_clip_max_norm=float(t["grad_clip_max_norm"]),
            cosine_T_0=int(t["scheduler"]["T_0"]),
            cosine_T_mult=int(t["scheduler"]["T_mult"]),
        )


@dataclass
class FoldResult:
    """Outputs of one gradient-fit / early-stop training run."""

    model: GRQNet
    best_val_macro_f1: float
    best_epoch: int
    val_predictions: np.ndarray   # argmax predictions on val
    val_probabilities: np.ndarray  # softmax probabilities on val
    val_group_weights: np.ndarray  # alpha weights on val (n, 4)
    n_epochs_trained: int


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool, seed: int
) -> DataLoader:
    """Create a DataLoader with a deterministic shuffle generator."""
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=generator
    )


@dataclass
class PredictionResult:
    """Predictions from a fixed GRQ-Net checkpoint."""

    predictions: np.ndarray
    probabilities: np.ndarray
    group_weights: np.ndarray


def predict_grqnet(model: GRQNet, X: np.ndarray) -> PredictionResult:
    """Evaluate a fixed checkpoint without changing model state."""
    model.eval()
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        logits, alpha = model(X_t, return_attention=True)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        weights = alpha.cpu().numpy()
    return PredictionResult(
        predictions=probs.argmax(axis=1),
        probabilities=probs,
        group_weights=weights,
    )


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    train_cfg: TrainConfig,
    model_cfg: GRQNetConfig | None = None,
    seed: int = 42,
    class_weight_labels: np.ndarray | None = None,
    verbose: bool = False,
) -> FoldResult:
    """Train GRQ-Net on one fold and return the best-checkpoint model.

    Parameters
    ----------
    X_train, y_train
        Training-fold UFT matrix (n_train, 50) and labels (n_train,).
        ``X_train`` must already be the output of the fold-specific
        :class:`src.features.uft.UFTPreprocessor` (i.e., scalers fit on
        train rows only).
    X_val, y_val
        Validation-fold UFT matrix and labels, transformed by the SAME
        fold preprocessor.
    train_cfg
        Training hyperparameters (Table 3).
    model_cfg
        GRQ-Net architecture config; defaults to Table 3 values.
    seed
        Primary neural-network seed.
    class_weight_labels
        Optional labels from the complete outer-training subset (or all 712
        final modelling rows). When provided, inverse-frequency class weights
        are computed from these rows rather than from the gradient subset.
    verbose
        Print per-epoch validation macro-F1.

    Returns
    -------
    FoldResult
    """
    # ---- Reproducibility: seed before model construction ------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GRQNet(model_cfg)

    # Manuscript protocol: class weights use the complete outer-training
    # subset (or all 712 rows for the final model), while gradient updates use
    # only the gradient-training rows after the internal 15% split.
    weight_labels = y_train if class_weight_labels is None else np.asarray(class_weight_labels, dtype=int)
    class_weights = compute_inverse_frequency_weights(
        weight_labels, n_classes=train_cfg.n_classes
    )
    criterion = SmoothedWeightedCrossEntropy(
        class_weights=class_weights,
        label_smoothing=train_cfg.label_smoothing,
        n_classes=train_cfg.n_classes,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=train_cfg.cosine_T_0, T_mult=train_cfg.cosine_T_mult
    )

    loader = _make_loader(
        X_train, y_train, train_cfg.batch_size, shuffle=True, seed=seed
    )

    X_val_t = torch.from_numpy(X_val.astype(np.float32))

    best_val_f1 = -1.0
    best_state: dict | None = None
    best_epoch = -1
    wait = 0
    epochs_run = 0

    for epoch in range(train_cfg.max_epochs):
        epochs_run = epoch + 1
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.grad_clip_max_norm
            )
            optimizer.step()
        scheduler.step()

        # ---- Validation: macro-F1 monitor --------------------------------
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_pred = val_logits.argmax(dim=1).numpy()
        val_f1 = compute_metrics(y_val, val_pred, n_classes=train_cfg.n_classes).macro_f1

        if verbose:
            print(f"      epoch {epoch + 1:3d}: val macro-F1 = {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= train_cfg.early_stop_patience:
                break

    # ---- Restore best checkpoint ------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = predict_grqnet(model, X_val)

    return FoldResult(
        model=model,
        best_val_macro_f1=best_val_f1,
        best_epoch=best_epoch,
        val_predictions=pred.predictions,
        val_probabilities=pred.probabilities,
        val_group_weights=pred.group_weights,
        n_epochs_trained=epochs_run,
    )
