"""Step 03 – Train and evaluate the classical UC1 baselines

This script trains the three TF-IDF models implemented in src/baselines.py using the UC1 training split.
It evaluates each model on the full test set of 13,030 emails and saves the evaluation metrics and predictions for every test email.
Step 04 reuses these predictions to compare the baselines with GPT-4o on the same emails.

Thesis: Chapter 4, Section 4.1 (Table 2 and Figure 2).

Reads:
READ_DIR/train.csv — training split created by 02_prepare_data.py.
READ_DIR/test.csv — test split created by 02_prepare_data.py.

Writes:
SAVE_DIR/uc1_baselines.csv — precision, recall, F1 score, accuracy, ROC-AUC and false-positive rate (FPR) for each model.
SAVE_DIR/uc1_predictions.csv — each model’s prediction and score for every test email.

Usage: python 03_run_uc1_baselines.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C
from src import baselines as B
from src import evaluation as E

# Folders used by this script (defaults from config.py; change here or in config.py).
READ_DIR = C.PROCESSED_DIR      # train.csv and test.csv (from 02)
SAVE_DIR = C.RESULTS_DIR        # result tables
SAVE_DIR.mkdir(exist_ok=True)

# Columns used by tfidf_lr_meta; the text-only models use "text" alone.
META_COLS = ["text", "sender_tld", "num_urls"]


def load_split(name: str) -> pd.DataFrame:
    """Load one UC1 split and fill blanks in the columns the models need."""
    df = pd.read_csv(READ_DIR / f"{name}.csv")
    for col in ("text", "sender_tld"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "num_urls" in df.columns:
        df["num_urls"] = pd.to_numeric(df["num_urls"], errors="coerce").fillna(0)
    return df


def main() -> int:
    """Train each baseline, score it on the test set and save the results."""
    # --- 1. Load the data ---
    print("Loading splits ...")
    train, test = load_split("train"), load_split("test")
    print(f"  train={len(train):,}  test={len(test):,}")
    y_test = test["label"].to_numpy(int)

    # --- 2. Train and evaluate each model ---
    rows, preds = [], {"label": y_test}
    for name, (builder, needs_meta) in B.MODELS.items():
        print(f"\n=== {name} ===")
        model = builder()
        Xtr = train[META_COLS] if needs_meta else train[["text"]]
        Xte = test[META_COLS] if needs_meta else test[["text"]]
        t0 = time.time()
        model.fit(Xtr, train["label"].to_numpy(int))
        fit_s = time.time() - t0
        y_pred = model.predict(Xte).astype(int)
        # A model without probabilities falls back to its hard labels as scores.
        try:
            scores = model.predict_proba(Xte)[:, 1]
        except Exception:
            scores = y_pred.astype(float)

        m = E.metrics(y_test, y_pred)
        # ROC-AUC is undefined when the scores or labels hold a single class.
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, scores)
        except Exception:
            auc = float("nan")

        print(f"  fit {fit_s:.1f}s  P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} Acc={m['accuracy']:.3f} AUC={auc:.3f} FPR={m['fpr']:.3f}")
        rows.append({"model": name, "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "accuracy": m["accuracy"], "roc_auc": auc,
                     "fpr": m["fpr"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                     "tn": m["tn"], "fit_seconds": round(fit_s, 1)})
        preds[name + "_pred"] = y_pred
        preds[name + "_score"] = np.round(scores, 4)

    # --- 3. Save and print the results ---
    results = pd.DataFrame(rows)
    results.to_csv(SAVE_DIR / "uc1_baselines.csv", index=False)
    pd.DataFrame(preds).to_csv(SAVE_DIR / "uc1_predictions.csv", index=False)

    print("\n================= UC1 BASELINE RESULTS =================")
    print(results[["model", "precision", "recall", "f1", "accuracy", "roc_auc", "fpr"]]
          .to_string(index=False))
    print(f"\nSaved -> {SAVE_DIR / 'uc1_baselines.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
