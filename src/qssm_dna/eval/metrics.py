"""Binary metrics (MCC, balanced accuracy, AUROC, average precision) and
the validation threshold rule. Average precision is sklearn's
average_precision_score, not a trapezoidal PR-AUC.
"""

from __future__ import annotations

import numpy as np


def binary_metrics(y_true, prob_pos, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (average_precision_score,
                                 balanced_accuracy_score,
                                 matthews_corrcoef, roc_auc_score)
    y_true = np.asarray(y_true)
    prob_pos = np.asarray(prob_pos)
    pred = (prob_pos >= threshold).astype(int)
    out = {
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, prob_pos))
        out["average_precision"] = float(
            average_precision_score(y_true, prob_pos))
    else:
        out["auroc"] = None
        out["average_precision"] = None
    return out


def pick_threshold(y_val, prob_val) -> float:
    """Threshold with the highest validation MCC over linspace(0.02, 0.98,
    97); the first maximum is kept."""
    from sklearn.metrics import matthews_corrcoef
    best_t, best_m = 0.5, -2.0
    for t in np.linspace(0.02, 0.98, 97):
        m = matthews_corrcoef(y_val, (prob_val >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t
