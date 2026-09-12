"""Shared PyTorch training loop for neural baselines.

Uses the same class-weighted smoothed cross-entropy, AdamW, warm-restart cosine
scheduler, batch size, gradient clipping, and macro-F1 early stopping as GRQ-Net.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.loss import SmoothedWeightedCrossEntropy, compute_inverse_frequency_weights
from src.training.metrics import ClassificationMetrics, compute_metrics
from src.training.trainer import TrainConfig


@dataclass
class NeuralBaselineResult:
    model: nn.Module
    best_val_macro_f1: float
    best_epoch: int
    n_epochs_trained: int


def _loader(X, y, batch_size, seed):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)


def predict_torch_classifier(model: nn.Module, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs.argmax(axis=1), probs


def train_neural_baseline(
    model: nn.Module,
    X_gradient: np.ndarray,
    y_gradient: np.ndarray,
    X_early: np.ndarray,
    y_early: np.ndarray,
    *,
    class_weight_labels: np.ndarray,
    train_cfg: TrainConfig,
    seed: int,
) -> NeuralBaselineResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = compute_inverse_frequency_weights(class_weight_labels, n_classes=train_cfg.n_classes)
    loss_fn = SmoothedWeightedCrossEntropy(weights, label_smoothing=train_cfg.label_smoothing, n_classes=train_cfg.n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=train_cfg.cosine_T_0, T_mult=train_cfg.cosine_T_mult)
    loader = _loader(X_gradient, y_gradient, train_cfg.batch_size, seed)
    Xev = torch.as_tensor(X_early, dtype=torch.float32)
    best = -1.0
    best_state = None
    best_epoch = -1
    wait = 0
    epochs_run = 0
    for epoch in range(train_cfg.max_epochs):
        epochs_run = epoch + 1
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_max_norm)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xev).argmax(dim=1).cpu().numpy()
        f1 = compute_metrics(y_early, pred, n_classes=train_cfg.n_classes).macro_f1
        if f1 > best:
            best = f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= train_cfg.early_stop_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return NeuralBaselineResult(model, best, best_epoch, epochs_run)


def evaluate_neural_baseline(result: NeuralBaselineResult, X: np.ndarray, y: np.ndarray) -> ClassificationMetrics:
    pred, _ = predict_torch_classifier(result.model, X)
    return compute_metrics(y, pred)
