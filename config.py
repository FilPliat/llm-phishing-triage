"""Shared settings for every script: folders, labels and data split.

Labels are 0 = legitimate and 1 = phishing; the random seed is 42.
To move data or results, change the paths in the FOLDER LOCATIONS block below.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# FOLDER LOCATIONS
# Change these lines to store files elsewhere. Absolute paths also work,
# for example Path(r"D:\phishing\results").
# ---------------------------------------------------------------------------
CODE_ROOT = Path(__file__).resolve().parent             # this folder (Thesis Codes)
DATASET_DIR = CODE_ROOT / "data"                       # raw public UC1 dataset files (input)
PROCESSED_DIR = CODE_ROOT / "data" / "processed"        # train/val/test splits and the UC2 dataset
RESULTS_DIR = CODE_ROOT / "results"                     # result tables, figures and the GPT-4o cache
ENV_FILE = CODE_ROOT / ".env"                           # file holding OPENAI_API_KEY

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Labels ---
LEGIT, PHISH = 0, 1
LABEL_NAMES = {LEGIT: "legitimate", PHISH: "phishing"}

# --- Raw UC1 dataset files (placed directly in DATASET_DIR) ---
KAGGLE_CSV = DATASET_DIR / "phishing_legit_dataset_KD_10000.csv"
ENRON_LABELED_CSV = DATASET_DIR / "enron_spam_data.csv"
ENRON_RAW_CSV = DATASET_DIR / "emails.csv"   # full CMU Enron (raw MIME; not needed for the results)

# Figshare CSVs, found by name under DATASET_DIR. True = phishing only; False = has a label column.
FIGSHARE_FILES = {
    "Nazario.csv":   True,     # gold-standard real phishing (phishing-only)
    "Nazario_5.csv": True,     # extended Nazario phishing corpus
    "CEAS-08.csv":   False,    # phishing + ham (has label column)
    "Assassin.csv":  False,    # SpamAssassin spam + ham (has label column)
    "Nigerian_5.csv": True,    # 419 fraud (optional)
    "Ling.csv":      False,    # Ling-Spam (optional)
}

# --- Split ---
SPLIT = {"train": 0.70, "val": 0.10, "test": 0.20}
RANDOM_SEED = 42                # fixed seed for reproducible splits and samples
BALANCE_CLASSES = True          # balance to about 50/50 phishing/legitimate (Section 3.5)

# --- Columns of the unified email record ---
# sender_domain, sender_tld and num_urls are derived offline (no live lookups).
DATA_COLS = ["text", "subject", "body", "sender", "urls",
             "sender_domain", "sender_tld", "num_urls",
             "label", "label_name", "source"]
SCHEMA = DATA_COLS + ["split"]
