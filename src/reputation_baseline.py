"""Two UC2 baselines that never read the email text: a reputation rule and a first-contact rule.

Both are expected to miss trusted-source phishing, because its links are clean and its sender is
a real contact. The thresholds are fixed and not tuned to the dataset.

Thesis: Section 3.4 (comparison models).
Used by: 08_run_uc2_evaluation.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DOMAIN_AGE_THRESHOLD_DAYS = 180   # flag if the link's domain is younger than about 6 months
DGA_SCORE_THRESHOLD = 0.5         # flag if the DGA-likelihood score is above this value
# Any VirusTotal detection (vt_malicious > 0) always flags.


def predict(df: pd.DataFrame) -> np.ndarray:
    """Flag emails whose link has a young domain, high DGA score or VirusTotal hit."""
    has_url = df["has_url"].astype(bool)
    age = pd.to_numeric(df["domain_age_days"], errors="coerce")
    dga = pd.to_numeric(df["dga_score"], errors="coerce").fillna(0.0)
    vt = pd.to_numeric(df["vt_malicious"], errors="coerce").fillna(0)

    young = has_url & age.notna() & (age < DOMAIN_AGE_THRESHOLD_DAYS)
    dga_hit = has_url & (dga > DGA_SCORE_THRESHOLD)
    vt_hit = has_url & (vt > 0)

    flag = young | dga_hit | vt_hit
    return flag.astype(int).to_numpy()


def predict_first_contact(df: pd.DataFrame) -> np.ndarray:
    """Flag emails whose sender has never written to the recipient before (1/0 array)."""
    prior_contact = df["prior_contact"].astype(bool)
    return (~prior_contact).astype(int).to_numpy()
