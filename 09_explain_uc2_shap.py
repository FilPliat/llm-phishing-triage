"""Step 09 – Explain three GPT-4o verdicts on UC2 emails with SHAP

GPT-4o is a black box behind an API, so SHAP masks words and asks the model again to see how each word moves P(phishing).
The script does this for three hand-picked UC2 emails, using the same prompt and settings as step 08.
Each email needs up to 300 GPT-4o calls. Run it from inside the Thesis Codes folder.

Thesis: Chapter 4, Section 4.3 (Figures 7–9).

Reads: DATASET_FILE — the UC2 dataset from 06 and 07.

Writes:
SAVE_DIR/shap_<tag>.png, .html, .txt — chart, word highlights and top tokens per email.
SAVE_DIR/_call_cache.json — cached GPT-4o scores.
SAVE_DIR/cost_summary.txt — API calls and estimated cost.

Usage: python 09_explain_uc2_shap.py

API cost: calls the OpenAI API; needs OPENAI_API_KEY in the .env file.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# llm_classifier is imported directly from src/, so src/ goes on the path too.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
SRC_DIR = C.CODE_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Folders used by this script (defaults from config.py; change here or in config.py).
DATASET_FILE = C.PROCESSED_DIR / "uc2_dataset.csv"  # UC2 dataset (from 06)
SAVE_DIR = C.RESULTS_DIR / "shap_explanations"      # SHAP outputs and call cache
ENV_FILE = C.ENV_FILE                               # .env file holding OPENAI_API_KEY

# Load .env before importing llm_classifier, which reads LLM_MODEL from the environment.
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
except ImportError:
    pass  # main() stops if OPENAI_API_KEY is missing.

from llm_classifier import (  # noqa: E402
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_UC2_TENANT_ADDENDUM,
    JSON_INSTRUCTION,
    DEFAULT_MODEL,
    TEMPERATURE,
    format_email,
)

# --- Settings ---
CACHE_PATH = SAVE_DIR / "_call_cache.json"   # cache of GPT-4o scores, keyed by text hash

USE_IMPROVED_PROMPT = True   # explain the improved prompt (Section 4.3); False = baseline prompt

# A fixed ceiling keeps the cost predictable; 300 gave a stable top-10 ranking in testing.
MAX_EVALS = 300   # maximum masked variants scored by GPT-4o per email

PRICE_IN_PER_M, PRICE_OUT_PER_M = 2.50, 10.00   # USD per 1M input / output tokens (as in 08)

# The three UC2 emails to explain: row index, file-name tag and why it was chosen.
EXAMPLES = [
    {
        "idx": 6,
        "tag": "idx6_typosquat",
        "note": (
            "trusted_source_phishing / typosquat_link — baseline prompt missed this, "
            "improved prompt caught it. authkey tenant string is a double-R typosquat "
            "of MERIDIANPARTNERS."
        ),
    },
    {
        "idx": 623,
        "tag": "idx623_brand_mismatch",
        "note": (
            "trusted_source_phishing / device_code — baseline prompt missed this, "
            "improved prompt caught it. Body claims 'SignRelay' branding; sender "
            "domain northridgestudio.com has no relation to that name."
        ),
    },
    {
        "idx": 1551,
        "tag": "idx1551_false_positive",
        "note": (
            "legitimate — genuinely correct sender/link match (bellwoodanalytics.com "
            "both places), but the improved prompt's added sensitivity to "
            "password-reset + urgency language flags it anyway (false positive)."
        ),
    },
]


def load_examples() -> pd.DataFrame:
    """Load the UC2 dataset that holds the example emails."""
    df = pd.read_csv(DATASET_FILE)
    return df


def build_text_block(row: pd.Series) -> str:
    """Rebuild the email text GPT-4o saw, without JSON_INSTRUCTION so SHAP never masks it."""
    return format_email(row)


# --- Scoring function (the black box that SHAP explains) ---

def _cache_key(text: str) -> str:
    """Return the SHA-256 hash of a text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CostTracker:
    """Count API calls and cache hits and estimate the token cost of a run."""

    def __init__(self):
        """Start all counters at zero."""
        self.calls = 0
        self.cache_hits = 0
        self.input_tokens_est = 0
        self.output_tokens_est = 0

    def add_call(self, prompt_chars: int, completion_chars: int):
        """Record one API call and add its estimated tokens."""
        self.calls += 1
        # About 4 characters per token; good enough for an estimate.
        self.input_tokens_est += prompt_chars // 4
        self.output_tokens_est += completion_chars // 4

    def estimated_cost_usd(self) -> float:
        """Return the estimated cost in US dollars."""
        return (self.input_tokens_est / 1e6) * PRICE_IN_PER_M + (
            self.output_tokens_est / 1e6
        ) * PRICE_OUT_PER_M

    def report(self) -> str:
        """Return a short summary of calls, tokens and cost."""
        return (
            f"API calls made: {self.calls} (cache hits saved: {self.cache_hits})\n"
            f"Estimated input tokens: ~{self.input_tokens_est:,}\n"
            f"Estimated output tokens: ~{self.output_tokens_est:,}\n"
            f"Estimated cost: ~${self.estimated_cost_usd():.2f} "
            f"(gpt-4o-2024-08-06 pricing, same table as 08_run_uc2_evaluation.py)"
        )


def make_scorer(cache: dict, tracker: CostTracker):
    """Build the function f(texts) -> P(phishing) array that SHAP calls on masked texts."""
    from openai import OpenAI

    client = OpenAI()
    system_content = SYSTEM_PROMPT + (
        SYSTEM_PROMPT_UC2_TENANT_ADDENDUM if USE_IMPROVED_PROMPT else ""
    )

    def score_one(text: str) -> float:
        """Return P(phishing) for one text, from the cache or one GPT-4o call."""
        key = _cache_key(text)
        if key in cache:
            tracker.cache_hits += 1
            return cache[key]

        user_content = text + "\n" + JSON_INSTRUCTION
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        # Heavily masked text can break JSON mode, so retry up to three times.
        last_err = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                data = json.loads(content)
                cls = str(data.get("classification", "legitimate")).lower()
                conf = float(data.get("confidence", 0.0))
                score = conf if cls.startswith("phish") else 1.0 - conf
                tracker.add_call(len(system_content) + len(user_content), len(content))
                cache[key] = score
                return score
            except Exception as e:  # noqa: BLE001 - retry on any error, but report it below
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        # Use a neutral 0.5 so one bad variant does not stop the run.
        print(f"  [warn] scoring failed after retries ({last_err}); using 0.5")
        cache[key] = 0.5
        return 0.5

    def f(texts) -> np.ndarray:
        """Score a batch of texts; this is the function SHAP calls."""
        return np.array([score_one(t) for t in texts])

    return f


def save_cache(cache: dict):
    """Write the score cache to CACHE_PATH."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as fh:
        json.dump(cache, fh)


def load_cache() -> dict:
    """Read the score cache from CACHE_PATH, or return an empty dict."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as fh:
            return json.load(fh)
    return {}


# --- Explanation and plots ---

def explain_one(row: pd.Series, tag: str, note: str, cache: dict, tracker: CostTracker):
    """Compute SHAP values for one email and save the text, chart and HTML views."""
    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text = build_text_block(row)
    print(f"\n=== {tag} ===\n{note}\n")
    print(f"Text block ({len(text)} chars):\n{text[:300]}...\n")

    masker = shap.maskers.Text(tokenizer=r"\W+")  # tokens are split at non-word characters
    f = make_scorer(cache, tracker)
    explainer = shap.Explainer(f, masker, algorithm="partition")

    shap_values = explainer([text], max_evals=MAX_EVALS, batch_size=10)
    save_cache(cache)

    sv = shap_values[0]
    tokens = list(sv.data)
    values = np.array(sv.values)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Ranked text summary (always works, unlike the HTML view) ---
    order = np.argsort(-np.abs(values))
    lines = [f"SHAP explanation — {tag}", note, "", f"Base value: {sv.base_values:.3f}",
             f"Predicted score (sum): {sv.base_values + values.sum():.3f}", "",
             "Top tokens by |impact on P(phishing)|:"]
    for i in order[:20]:
        tok = tokens[i].strip()
        if not tok:
            continue
        direction = "+phishing" if values[i] > 0 else "-phishing"
        lines.append(f"  {values[i]:+.4f}  {direction:11s}  {tok!r}")
    txt_path = SAVE_DIR / f"shap_{tag}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"  wrote {txt_path}")

    # --- 2. Bar chart of the top tokens ---
    top_n = 12
    top_idx = order[:top_n]
    top_idx = top_idx[np.argsort(values[top_idx])]  # sorted by value for a clean bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in values[top_idx]]
    ax.barh([tokens[i].strip() or "·" for i in top_idx], values[top_idx], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (push toward phishing →  |  ← push toward legitimate)")
    ax.set_title(f"GPT-4o token attributions — {tag}\n{note}", fontsize=9, wrap=True)
    fig.tight_layout()
    png_path = SAVE_DIR / f"shap_{tag}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {png_path}")

    # --- 3. HTML word-highlight view ---
    # The renderer varies by shap version, so a failure is only a warning.
    try:
        html = shap.plots.text(shap_values, display=False)
        html_path = SAVE_DIR / f"shap_{tag}.html"
        html_path.write_text(html)
        print(f"  wrote {html_path}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not render HTML text view ({e}); .png and .txt still have the data")


def main():
    """Explain the three example emails and save the cache and the cost summary."""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — load your .env first (same as 08_run_uc2_evaluation.py).")
        sys.exit(1)

    df = load_examples()
    cache = load_cache()
    tracker = CostTracker()

    for ex in EXAMPLES:
        row = df.iloc[ex["idx"]]
        explain_one(row, ex["tag"], ex["note"], cache, tracker)

    save_cache(cache)
    report = tracker.report()
    print("\n" + report)
    (SAVE_DIR / "cost_summary.txt").write_text(report)


if __name__ == "__main__":
    main()
