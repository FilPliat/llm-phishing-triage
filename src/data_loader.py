"""Load the public UC1 email datasets and merge them into one clean table.

Each source is converted to the shared columns in config.DATA_COLS, then deduplicated, split
70/10/20 by label and balanced to about 50/50 per split. The UC2 build reuses these helpers.

Thesis: Section 3.5 (UC1 dataset).
Used by: 01_smoke_test.py, 02_prepare_data.py, 06_build_uc2_dataset.py.
"""
from __future__ import annotations
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C

# Email address (group 1 is the domain) and http(s) link patterns.
_EMAIL_RE = re.compile(r"[\w.+-]+@([A-Za-z0-9-]+\.[A-Za-z0-9.-]+)")
_URL_RE = re.compile(r"https?://[^\s\"'>)\]]+")


# --- Derived signal helpers ---------------------------------------------------
def sender_domain(sender: str) -> str:
    """Return the lowercase domain of a sender field, or "" if there is none."""
    if not isinstance(sender, str):
        return ""
    m = _EMAIL_RE.search(sender)
    if m:
        return m.group(1).lower().strip(".")
    # Fallback for malformed addresses that still contain an "@".
    m2 = re.search(r"@([A-Za-z0-9.-]+)", sender)
    return m2.group(1).lower().strip(".") if m2 else ""


def sender_tld(domain: str) -> str:
    """Return the top-level domain of a domain name, or "" if there is none."""
    return domain.rsplit(".", 1)[-1] if domain and "." in domain else ""


def count_urls(urls_field: str, body: str) -> int:
    """Count the links of an email, preferring the source's urls field over the body."""
    if isinstance(urls_field, str) and urls_field.strip():
        found = _URL_RE.findall(urls_field)
        if found:
            return len(found)
        # Some sources list bare links without a scheme.
        parts = [u for u in re.split(r"[\s,;]+", urls_field.strip()) if u]
        if parts:
            return len(parts)
    return len(_URL_RE.findall(body or ""))


def _finalize(df: pd.DataFrame, source: str, label=None) -> pd.DataFrame:
    """Convert a raw frame to config.DATA_COLS and drop rows with empty text."""
    df = df.copy()
    for col in ("subject", "body", "sender", "urls"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if label is not None:
        df["label"] = label
    df["label"] = df["label"].astype(int)
    df["text"] = (df["subject"].str.strip() + "\n\n" + df["body"].str.strip()).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["sender_domain"] = df["sender"].map(sender_domain)
    df["sender_tld"] = df["sender_domain"].map(sender_tld)
    df["num_urls"] = [count_urls(u, b) for u, b in zip(df["urls"], df["body"])]
    df["label_name"] = df["label"].map(C.LABEL_NAMES)
    df["source"] = source
    return df[C.DATA_COLS].reset_index(drop=True)


def _find(filename: str) -> Path | None:
    """Find a file by name (case-insensitive) under config.DATASET_DIR."""
    if not C.DATASET_DIR.exists():
        return None
    for p in C.DATASET_DIR.rglob("*"):
        if p.is_file() and p.name.lower() == filename.lower():
            return p
    return None


# --- Individual loaders -------------------------------------------------------
def load_kaggle() -> pd.DataFrame | None:
    """Load the Kaggle 2026 dataset, or return None if the file is missing."""
    if not C.KAGGLE_CSV.exists():
        warnings.warn(f"Kaggle CSV not found: {C.KAGGLE_CSV}")
        return None
    df = pd.read_csv(C.KAGGLE_CSV)   # columns: text, label, phishing_type, severity, confidence
    out = pd.DataFrame({"body": df["text"], "label": df["label"]})
    return _finalize(out, "kaggle_2026")


def load_enron_labeled(include_spam: bool = False) -> pd.DataFrame | None:
    """Load the Enron ham emails (plus spam as phishing if include_spam), or None if missing."""
    if not C.ENRON_LABELED_CSV.exists():
        warnings.warn(f"Enron labeled CSV not found: {C.ENRON_LABELED_CSV}")
        return None
    df = pd.read_csv(C.ENRON_LABELED_CSV)   # columns: Message ID, Subject, Message, Spam/Ham, Date
    is_ham = df["Spam/Ham"].str.lower().eq("ham")
    base = pd.DataFrame({"subject": df["Subject"], "body": df["Message"]})
    ham = _finalize(base[is_ham], "enron_ham", label=C.LEGIT)
    if not include_spam:
        return ham
    spam = _finalize(base[~is_ham], "enron_spam", label=C.PHISH)
    return pd.concat([ham, spam], ignore_index=True)


def load_figshare_file(filename: str, phishing_only: bool) -> pd.DataFrame | None:
    """Load one Figshare CSV, or return None if it is not found.

    Args:
        filename: File name to find under config.DATASET_DIR.
        phishing_only: True if every row is phishing (the file has no label column).
    """
    path = _find(filename)
    if path is None:
        return None
    df = pd.read_csv(path)
    lc = {c.lower(): c for c in df.columns}
    out = pd.DataFrame({
        "subject": df[lc["subject"]] if "subject" in lc else "",
        "body": df[lc["body"]] if "body" in lc else "",
        "sender": df[lc["sender"]] if "sender" in lc else "",
        "urls": df[lc["urls"]] if "urls" in lc else "",
    })
    src = path.stem.lower()
    if phishing_only:
        return _finalize(out, src, label=C.PHISH)
    if "label" not in lc:
        warnings.warn(f"{filename}: no label column; treating all as phishing")
        return _finalize(out, src, label=C.PHISH)
    raw = df[lc["label"]]
    if raw.dtype == object:
        label = raw.str.lower().map(
            {"phishing": 1, "phish": 1, "spam": 1, "1": 1,
             "legitimate": 0, "legit": 0, "ham": 0, "0": 0}
        ).fillna(0).astype(int)
    else:
        label = (raw > 0).astype(int)
    out["label"] = label.values
    return _finalize(out, src)


def load_all_figshare() -> list[pd.DataFrame]:
    """Load every file in config.FIGSHARE_FILES, skipping missing ones."""
    out = []
    for name, phishing_only in C.FIGSHARE_FILES.items():
        df = load_figshare_file(name, phishing_only)
        if df is not None:
            out.append(df)
    return out


# --- Merge and split ----------------------------------------------------------
def build_unified(include_enron_spam: bool = False) -> pd.DataFrame:
    """Merge all UC1 sources into one table and drop duplicate texts."""
    parts = []
    for loader in (load_kaggle, lambda: load_enron_labeled(include_enron_spam)):
        df = loader()
        if df is not None:
            parts.append(df)
    parts.extend(load_all_figshare())
    if not parts:
        raise RuntimeError("No UC1 datasets found in the data folder (DATASET_DIR). See HOW_TO_RUN.txt.")
    unified = pd.concat(parts, ignore_index=True)
    unified = unified.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return unified


def stratified_split(df: pd.DataFrame, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Add a train/val/test "split" column (config.SPLIT), stratified by label."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""
    for label, grp in df.groupby("label"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * C.SPLIT["train"])
        n_va = int(n * C.SPLIT["val"])
        df.loc[idx[:n_tr], "split"] = "train"
        df.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
        df.loc[idx[n_tr + n_va:], "split"] = "test"
    return df


def balance_classes(df: pd.DataFrame, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Undersample the majority class in each split to about 50/50 and shuffle."""
    rng = np.random.default_rng(seed)
    out = []
    for split, grp in df.groupby("split"):
        counts = grp["label"].value_counts()
        n = counts.min()
        for label, sub in grp.groupby("label"):
            take = sub.sample(n=n, random_state=int(rng.integers(1e9))) if len(sub) > n else sub
            out.append(take)
    return pd.concat(out, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    # Quick check: table size per source and label.
    u = build_unified()
    print(f"Unified rows: {len(u):,}")
    print(u["source"].value_counts())
    print(u["label_name"].value_counts())
