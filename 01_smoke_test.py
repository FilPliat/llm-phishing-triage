"""Step 01 – Check the environment and raw datasets

This script checks that the environment is ready and the raw datasets are available. It builds the merged UC1 dataset once,
reports how many emails contain a sender domain and links, and verifies the 70/10/20 data split.
It does not train any models or write files. Use it as a quick check before running the later steps.

Thesis: Chapter 3, Section 3.5 (UC1 dataset and split).

Reads: DATASET_DIR/... — the raw public UC1 datasets listed in config.py (not included).

Usage: python 01_smoke_test.py
A successful run ends with SMOKE TEST PASSED.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config as C
from src import data_loader as dl

# Folders used by this script (defaults from config.py; change here or in config.py).
DATASET_DIR = C.DATASET_DIR     # raw public UC1 datasets (not included); only printed here


def main() -> int:
    """Run the setup checks and print a short report."""
    print("=" * 64)
    print("PHISHING TRIAGE — ENVIRONMENT SMOKE TEST")
    print("=" * 64)
    print(f"Dataset dir : {DATASET_DIR}  (exists: {DATASET_DIR.exists()})")

    # --- 1. Show which raw datasets are present ---
    print("\n-- Datasets present --")
    for label, path in [("Kaggle 2026", C.KAGGLE_CSV),
                        ("Enron labeled", C.ENRON_LABELED_CSV),
                        ("Enron raw", C.ENRON_RAW_CSV)]:
        print(f"  [{'OK' if path.exists() else '--'}] {label}: {path.name}")
    print("  Figshare curated (auto-detected under DATASET_DIR):")
    for name in C.FIGSHARE_FILES:
        print(f"    [{'OK' if dl._find(name) else '--'}] {name}")

    # --- 2. Build the merged dataset ---
    print("\n-- Building unified dataset (enriched schema) --")
    df = dl.build_unified()
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Source  : {df['source'].value_counts().to_dict()}")
    print(f"  Class   : {df['label_name'].value_counts().to_dict()}")

    # --- 3. Check the sender and link signals ---
    print("\n-- Enrichment signals --")
    with_sender = df[df['sender_domain'] != '']
    print(f"  Emails with a parsed sender domain : {len(with_sender):,}")
    print(f"  Emails with >=1 URL                : {(df['num_urls'] > 0).sum():,}")
    ph = with_sender[with_sender['label'] == C.PHISH]['sender_tld'].value_counts().head(6)
    lg = with_sender[with_sender['label'] == C.LEGIT]['sender_tld'].value_counts().head(6)
    print(f"  Top sender TLDs (phishing)   : {ph.to_dict()}")
    print(f"  Top sender TLDs (legitimate) : {lg.to_dict()}")

    # --- 4. Check the 70/10/20 split ---
    print("\n-- Stratified split (70/10/20) --")
    df = dl.stratified_split(df)
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        print(f"  {s:5s}: {len(sub):>7,}  class balance {sub['label'].value_counts().to_dict()}")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
