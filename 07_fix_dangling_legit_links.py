"""Step 07 – Repair legitimate UC2 emails that mention a missing link

GPT-4o sometimes wrote "see the link below" in a legitimate email that has no link.
That would be an artificial clue, so this script regenerates the subject and body of those emails.
Run it after 06 and before 08. It is safe to rerun.

Thesis: Chapter 3, Section 3.6 (quality control of the generated emails).

Reads: DATA_DIR/uc2_dataset.csv — the UC2 dataset from 06_build_uc2_dataset.py.

Writes: DATA_DIR/uc2_dataset.csv — the same file, with the affected rows repaired.

Usage: python 07_fix_dangling_legit_links.py

API cost: calls the OpenAI API; needs OPENAI_API_KEY in the .env file.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C
import importlib.util

# Reuse step 06; its name starts with a digit, so load it by file path.
_spec = importlib.util.spec_from_file_location(
    "build_uc2_dataset", Path(__file__).resolve().parent / "06_build_uc2_dataset.py")
BD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BD)

# Folders used by this script (defaults from config.py; change here or in config.py).
DATA_DIR = C.PROCESSED_DIR      # uc2_dataset.csv (from 06); patched in place
ENV_FILE = C.ENV_FILE           # .env file holding OPENAI_API_KEY

# Broad on purpose: any phrasing that implies a link or attachment counts.
DANGLING_RE = re.compile(   # text that refers to a link or attached document
    r"\blink\b|\bdocument (below|attached|provided)\b|attached (document|file)|see attached",
    re.IGNORECASE,
)


def find_dangling(df: pd.DataFrame) -> pd.Index:
    """Return the index of legitimate emails that mention a link but have no URL."""
    legit = df[df["scenario"] == "legitimate"]
    has_url = legit["urls"].fillna("").str.len() > 0
    mentions = legit["body"].fillna("").apply(lambda b: bool(DANGLING_RE.search(b)))
    return legit.index[(~has_url) & mentions]


def main() -> int:
    """Regenerate the affected emails and save the patched dataset in place."""
    # --- 1. Load the dataset and find the affected rows ---
    path = DATA_DIR / "uc2_dataset.csv"
    if not path.exists():
        print(f"{path} not found -- run 06_build_uc2_dataset.py first.")
        return 1
    df = pd.read_csv(path)

    bad_idx = find_dangling(df)
    print(f"Found {len(bad_idx)} legitimate row(s) with a dangling link/document reference "
          f"(body implies a link/attachment but 'urls' is empty).")
    if len(bad_idx) == 0:
        print("Nothing to fix.")
        return 0

    # --- 2. Regenerate each affected email ---
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
    except Exception:
        pass
    from openai import OpenAI
    client = OpenAI()

    model = "gpt-4o-2024-08-06"
    fixed, unresolved = 0, []
    for n, idx in enumerate(bad_idx):
        row = df.loc[idx]
        company_name = row.get("_partner_company", "") or "Contoso"
        request_type = row.get("_request_type", "a routine business update")
        urgency = row.get("_urgency", "none")
        ok = False
        for attempt in range(3):
            msgs = BD.build_gen_prompt("legitimate", company_name, request_type, urgency,
                                        technique="", wants_link=False)
            data = BD.call_llm(client, msgs, model)
            body = (data.get("body") or "").strip()
            subj = (data.get("subject") or "").strip()
            if body and not DANGLING_RE.search(body):
                df.at[idx, "subject"] = subj
                df.at[idx, "body"] = body
                # Rebuild "text" as in data_loader._finalize; the classifiers read this field.
                df.at[idx, "text"] = f"{subj}\n\n{body}".strip()
                fixed += 1
                ok = True
                break
        if not ok:
            unresolved.append(idx)
        print(f"\r  fixing: {n+1}/{len(bad_idx)}", end="", flush=True)
    print()

    # --- 3. Save and recheck ---
    df.to_csv(path, index=False)
    print(f"Patched {fixed}/{len(bad_idx)} row(s) -> {path}")
    if unresolved:
        print(f"{len(unresolved)} row(s) still non-compliant after 3 attempts each -- "
              f"left as-is, flagged for manual review: index {unresolved}")
    remaining = find_dangling(pd.read_csv(path))
    print(f"Remaining dangling rows after patch: {len(remaining)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
