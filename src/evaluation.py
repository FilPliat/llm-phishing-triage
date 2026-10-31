"""Confusion matrix, precision, recall, F1, accuracy and FPR, shared by every model.

FPR is reported because, in a SOC, a false alarm costs analyst time just like a missed phish.

Thesis: Section 3.8 (evaluation metrics).
Used by: 03_run_uc1_baselines.py, 04_run_uc1_llm.py, 08_run_uc2_evaluation.py.
"""
from __future__ import annotations
import numpy as np


def confusion(y_true, y_pred) -> dict:
    """Count tp, tn, fp and fn, with label 1 (phishing) as the positive class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics(y_true, y_pred) -> dict:
    """Return precision, recall, f1, accuracy, fpr and the counts (0.0 if a denominator is 0)."""
    c = confusion(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "fpr": fpr, **c}


def report(name: str, y_true, y_pred) -> dict:
    """Compute metrics(), print them on one line and return them."""
    m = metrics(y_true, y_pred)
    print(f"[{name}]  P={m['precision']:.3f}  R={m['recall']:.3f}  "
          f"F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}  FPR={m['fpr']:.3f}  "
          f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")
    return m

# ROC-AUC needs scores, not labels, so the runner scripts compute it with scikit-learn.
