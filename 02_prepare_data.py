"""Step 02 – Build and split the shared UC1 dataset

This script merges the UC1 source datasets using src/data_loader.py, removes duplicates and balances the classes.
It then creates the shared 70/10/20 train, validation and test split used by all UC1 models.
This ensures that every model is evaluated on the same test emails.

Thesis: Chapter 3, Section 3.5 (UC1 dataset and the 70/10/20 split).

Reads:
config.DATASET_DIR/... — the raw public UC1 datasets, loaded by src/data_loader.py.

Writes:
SAVE_DIR/unified.csv — all balanced emails, including a split column.
SAVE_DIR/train.csv — 70% training split.
SAVE_DIR/val.csv — 10% validation split.
SAVE_DIR/test.csv — 20% test split.

Usage: python 02_prepare_data.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config as C
from src import data_loader as dl

# Folders used by this script (defaults from config.py; change here or in config.py).
SAVE_DIR = C.PROCESSED_DIR      # unified.csv and the train/val/test splits


def main() -> int:
    """Build the merged dataset, split it and save the CSV files."""
    print("Building unified dataset ...")
    df = dl.build_unified()
    df = dl.stratified_split(df)
    if C.BALANCE_CLASSES:
        df = dl.balance_classes(df)
        # Balancing drops rows, so split again to keep the 70/10/20 ratios exact.
        df = dl.stratified_split(df)

    out = SAVE_DIR
    df.to_csv(out / "unified.csv", index=False)
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        sub.to_csv(out / f"{s}.csv", index=False)
        print(f"  {s:5s}: {len(sub):>7,}  -> {out / (s + '.csv')}")
    print(f"\nWrote unified dataset ({len(df):,} rows) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
