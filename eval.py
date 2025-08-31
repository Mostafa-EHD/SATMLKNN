"""
eval.py — Prequential (test-then-train) evaluation for streaming multi-label classifiers.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any

from metrics import subset_accuracy, hamming_loss, precision_recall_f1

class PrequentialEvaluator:
    """
    Progressive validation (a.k.a. test-then-train).
    Models must implement:
      - predict_one(x): dict[label -> bool] OR dict[label -> {True: p, False: q}]
      - learn_one(x, y_vec): update model with sample
    """

    def __init__(self, models: Dict[str, Any], label_names, report_every: int = 1000):
        self.models = models
        self.labels = list(label_names)
        self.L = len(self.labels)
        self.report_every = int(report_every)

    # ---------- helpers ----------
    def _dict_bool_to_vec(self, d: dict) -> np.ndarray:
        return np.array([1 if d.get(lbl, False) else 0 for lbl in self.labels], dtype=int)

    def _proba_to_vec(self, d: dict, thr: float = 0.5) -> np.ndarray:
        return np.array([1 if d[l].get(True, 0.0) >= thr else 0 for l in self.labels], dtype=int)

    # ---------- main ----------
    def run(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        results = {name: {"y_true": [], "y_pred": [], "time_sec": 0.0} for name in self.models}
        n = len(X)

        for i in range(n):
            x_i = X[i]
            y_vec = Y[i].astype(int) if not isinstance(Y[i], dict) else self._dict_bool_to_vec(Y[i])

            for name, model in self.models.items():
                t0 = time.perf_counter()
                pred = model.predict_one(x_i)
                t1 = time.perf_counter()

                # Handle output format
                if isinstance(pred, dict) and len(pred) and isinstance(next(iter(pred.values())), dict):
                    thr = getattr(model, "inference_threshold", 0.5)
                    y_hat = self._proba_to_vec(pred, thr=thr)
                elif isinstance(pred, dict):
                    y_hat = self._dict_bool_to_vec(pred)
                else:
                    y_hat = np.asarray(pred, dtype=int)

                results[name]["y_true"].append(y_vec)
                results[name]["y_pred"].append(y_hat)
                results[name]["time_sec"] += (t1 - t0)

                # then learn
                model.learn_one(x_i, y_vec)

            if verbose and (i + 1) % self.report_every == 0:
                print(f"[{i+1}/{n}] processed")

        # collapse results + compute metrics
        summary = {}
        for name, d in results.items():
            y_true = np.vstack(d["y_true"])
            y_pred = np.vstack(d["y_pred"])
            sa = subset_accuracy(y_true, y_pred)
            hl = hamming_loss(y_true, y_pred)
            p_micro, r_micro, f_micro = precision_recall_f1(y_true, y_pred, "micro")
            p_macro, r_macro, f_macro = precision_recall_f1(y_true, y_pred, "macro")
            summary[name] = {
                "subset_acc": sa,
                "hamming_loss": hl,
                "micro_f1": f_micro,
                "macro_f1": f_macro,
                "time_per_sample_ms": 1000.0 * d["time_sec"] / n,
            }
        return summary
