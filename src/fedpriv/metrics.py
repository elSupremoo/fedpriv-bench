"""Evaluation metrics: accuracy, macro-F1, per-class recall."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score


@torch.no_grad()
def predict(model, X: torch.Tensor, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = [model(X[i : i + batch]).argmax(1) for i in range(0, len(X), batch)]
    return torch.cat(out).cpu().numpy()


def scores(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict:
    labels = list(range(num_classes))
    return {
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": [
            float(r)
            for r in recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        ],
    }


def evaluate(model, X: torch.Tensor, y_np: np.ndarray, num_classes: int = 3) -> dict:
    return scores(y_np, predict(model, X), num_classes)
