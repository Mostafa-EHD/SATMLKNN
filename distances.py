"""
distances.py — common distance functions + a small registry.

All functions return a *distance* (smaller = closer).
Cosine is returned as 1 - cosine_similarity.
Jensen–Shannon is returned as the metric form: sqrt(JS divergence).
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict

EPS = 1e-12

def _as_np(x) -> np.ndarray:
    """Force numeric numpy array."""
    return np.asarray(x, dtype=float)

def manhattan(u, v) -> float:
    u, v = _as_np(u), _as_np(v)
    return float(np.abs(u - v).sum())

def euclidean(u, v) -> float:
    u, v = _as_np(u), _as_np(v)
    return float(np.sqrt(np.square(u - v).sum()))

def minkowski(u, v, p: float = 3.0) -> float:
    u, v = _as_np(u), _as_np(v)
    # Standard Lp (no weird “sum then 1/p” shortcut crap)
    return float(np.power(np.abs(u - v).sum() + EPS, 1.0 / p))

def cosine(u, v) -> float:
    """
    Cosine *distance* = 1 - cosine_similarity.
    Safe against zero vectors.
    """
    u, v = _as_np(u), _as_np(v)
    num = float(np.dot(u, v))
    den = float(np.linalg.norm(u) * np.linalg.norm(v) + EPS)
    return 1.0 - (num / den)

def hamming_feat(u, v) -> float:
    """
    Feature-wise Hamming distance in [0,1] for binary/categorical features.
    If features aren’t binary, it’s still a simple inequality rate.
    """
    u, v = _as_np(u), _as_np(v)
    if u.size == 0:  # avoid division by zero if someone passes empty vectors
        return 0.0
    return float(np.not_equal(u, v).mean())

def jensen_shannon(p, q) -> float:
    """
    Jensen–Shannon distance over non-negative vectors treated as distributions.
    Returns sqrt(JS divergence), which is a proper metric in [0, sqrt(log 2)].
    """
    p, q = _as_np(p), _as_np(q)
    # Make non-negative and normalize
    p = np.clip(p, 0.0, None); q = np.clip(q, 0.0, None)
    ps = p.sum(); qs = q.sum()
    if ps <= 0 and qs <= 0:
        return 0.0
    if ps <= 0: p = np.ones_like(p); ps = p.sum()
    if qs <= 0: q = np.ones_like(q); qs = q.sum()
    p = p / (ps + EPS)
    q = q / (qs + EPS)
    m = 0.5 * (p + q)

    def _kl(a, b):
        a = np.clip(a, EPS, None)
        b = np.clip(b, EPS, None)
        return np.sum(a * np.log(a / b))

    js_div = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(js_div))

REGISTRY: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "minkowski": lambda u, v: minkowski(u, v, p=3.0),
    "cosine": cosine,
    "hamming": hamming_feat,
    "js": jensen_shannon,
}
