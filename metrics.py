"""
metrics.py — evaluation metrics for multi-label classification streams.
Implements subset accuracy, Hamming loss, micro-F1, macro-F1.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Exact match: proportion of samples where *all* labels match.
    """
    return float(np.mean(np.all(y_true == y_pred, axis=1)))

def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of labels misclassified.
    """
    return float(np.mean(y_true != y_pred))

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                        average: str = "micro") -> Tuple[float, float, float]:
    """
    Precision, recall, F1 for multi-label classification.
    average: "micro" or "macro".
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    if average == "micro":
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        return prec, rec, f1

    if average == "macro":
        L = y_true.shape[1]
        precs, recs, f1s = [], [], []
        for l in range(L):
            t, p = y_true[:, l], y_pred[:, l]
            tp = int(((t == 1) & (p == 1)).sum())
            fp = int(((t == 0) & (p == 1)).sum())
            fn = int(((t == 1) & (p == 0)).sum())
            pr = tp / (tp + fp + 1e-12)
            rc = tp / (tp + fn + 1e-12)
            f1 = 2 * pr * rc / (pr + rc + 1e-12)
            precs.append(pr); recs.append(rc); f1s.append(f1)
        return float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))

    raise ValueError("average must be 'micro' or 'macro'")
