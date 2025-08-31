"""
mlknn.py — Online ML-KNN baseline with sliding window and Laplace-smoothed priors.

Implements:
- predict_proba_one(x): returns {label: {True: p, False: q}}
- predict_one(x): MAP per label (p(True) >= p(False))
- learn_one(x, y_vec): updates window and prior counts

Notes:
- Distance comes from distances.REGISTRY (smaller = closer).
- Membership count uses unweighted k-NN votes (standard ML-KNN online variant).
"""

from __future__ import annotations
import numpy as np
from collections import deque, Counter
from typing import Sequence, Dict, List

from distances import REGISTRY as DIST


class MLKNNOnline:
    """
    Minimal, correct online ML-KNN with sliding window.
    """

    def __init__(
        self,
        label_names: Sequence[str],
        k: int = 5,
        window_size: int = 1000,
        distance: str = "manhattan",
        smoothing: float = 1.0,
    ):
        self.labels: List[str] = list(label_names)
        self.L: int = len(self.labels)
        self.k: int = int(k)
        self.window_size: int = int(window_size)
        if distance not in DIST:
            raise ValueError(f"Unknown distance '{distance}'. Choose from {list(DIST.keys())}.")
        self.distance = DIST[distance]
        self.smoothing: float = float(smoothing)

        # circular buffers
        self.X: deque = deque(maxlen=self.window_size)  # each x is np.ndarray (d,)
        self.Y: deque = deque(maxlen=self.window_size)  # each y is np.ndarray (L,) in {0,1}

        # frequencies for priors P(H_l=True)
        self.freq_pos: Counter = Counter()
        self.freq_neg: Counter = Counter()

        # used by evaluator when converting proba -> hard labels (if needed)
        self.inference_threshold: float = 0.5

    # ---------- internals ----------
    def _knn_query(self, x: np.ndarray):
        """Return indices and distances of k nearest neighbors among window."""
        n = len(self.X)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        dists = np.empty(n, dtype=float)
        for i, xi in enumerate(self.X):
            dists[i] = self.distance(x, xi)
        idx = np.argsort(dists)[: self.k]
        return idx, dists[idx]

    def _compute_priors(self) -> Dict[str, Dict[bool, float]]:
        """Laplace-smoothed priors for each label: P(H_l=True/False)."""
        priors: Dict[str, Dict[bool, float]] = {}
        for l in self.labels:
            pos = self.freq_pos[l]
            neg = self.freq_neg[l]
            p_pos = (pos + self.smoothing) / (pos + neg + 2.0 * self.smoothing)
            priors[l] = {True: p_pos, False: 1.0 - p_pos}
        return priors

    def _compute_membership_counts(self, nn_idx: np.ndarray):
        """For each label, count positive labels among k neighbors."""
        counts = {l: 0 for l in self.labels}
        for j in nn_idx:
            yj = self.Y[j]
            # yj is (L,), 0/1
            for li, l in enumerate(self.labels):
                counts[l] += int(yj[li] == 1)
        return counts

    def _compute_posteriors(self, priors, counts):
        """
        Simple posterior approximation per label using membership count c in [0..k].
        P(H_l=True | c) ∝ P(H_l=True) * P(c | H_l=True)
        with smoothed likelihoods:
           P(c | H=True) = (c + s) / (k + 2s)
           P(c | H=False) = (k - c + s) / (k + 2s)
        """
        post = {}
        k = self.k
        s = self.smoothing
        for l in self.labels:
            c = counts[l]
            p_c_pos = (c + s) / (k + 2.0 * s)
            p_c_neg = (k - c + s) / (k + 2.0 * s)
            p_pos = priors[l][True] * p_c_pos
            p_neg = priors[l][False] * p_c_neg
            Z = p_pos + p_neg + 1e-12
            post[l] = {True: p_pos / Z, False: p_neg / Z}
        return post

    # ---------- public API ----------
    def predict_proba_one(self, x: np.ndarray) -> Dict[str, Dict[bool, float]]:
        if len(self.X) == 0:
            # neutral when empty
            return {l: {True: 0.5, False: 0.5} for l in self.labels}
        x = np.asarray(x, dtype=float)

        priors = self._compute_priors()
        nn_idx, _ = self._knn_query(x)
        counts = self._compute_membership_counts(nn_idx)
        return self._compute_posteriors(priors, counts)

    def predict_one(self, x: np.ndarray) -> Dict[str, bool]:
        post = self.predict_proba_one(x)
        # MAP per label
        return {l: (post[l][True] >= post[l][False]) for l in self.labels}

    def learn_one(self, x: np.ndarray, y_vec: np.ndarray):
        x = np.asarray(x, dtype=float)
        y_vec = np.asarray(y_vec, dtype=int)
        if y_vec.shape[0] != self.L:
            raise ValueError(f"y_vec length {y_vec.shape[0]} != number of labels {self.L}")

        # update buffers
        self.X.append(x)
        self.Y.append(y_vec)

        # update prior counts
        for li, l in enumerate(self.labels):
            if y_vec[li] == 1:
                self.freq_pos[l] += 1
            else:
                self.freq_neg[l] += 1
        return self
