"""Step 04 – Run GPT-4o on the UC1 sample

Draws a balanced 400-email sample from the UC1 test set and classifies it with GPT-4o
using zero-shot, few-shot and chain-of-thought prompts (src/llm_classifier.py).
It records metrics, cost and latency, and scores the step 03 baselines on the same emails.
Answers are cached, so a rerun does not repeat finished calls.

Thesis: Chapter 4, Section 4.1 (Table 3, Figure 2).

Reads:
READ_DIR/test.csv — test split created by 02_prepare_data.py.
READ_DIR/train.csv — source of the few-shot examples.
SAVE_DIR/uc1_predictions.csv — baseline predictions from 03_run_uc1_baselines.py (optional).

Writes:
SAVE_DIR/uc1_llm_<n>.csv — metrics, cost and latency per strategy (uc1_llm_smoke.csv with --smoke).
CACHE_DIR/<model>__<strategy>.json — cached GPT-4o answers.

Usage: python 04_run_uc1_llm.py --n 400

Options:
--n — sample size (default 400).
--model — OpenAI model (default gpt-4o-2024-08-06).
--strategies — any of zero_shot, few_shot, chain_of_thought.
--smoke — 6-email test run with gpt-4o-mini.

API cost: calls the OpenAI API; needs OPENAI_API_KEY in the .env file.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C
from src import evaluation as E
from src import llm_classifier as L

# Folders used by this script (defaults from config.py; change here or in config.py).
READ_DIR = C.PROCESSED_DIR                  # train.csv and test.csv (from 02)
SAVE_DIR = C.RESULTS_DIR                    # result tables; uc1_predictions.csv (from 03)
CACHE_DIR = C.RESULTS_DIR / "llm_cache"     # cached GPT-4o answers
ENV_FILE = C.ENV_FILE                       # .env file holding OPENAI_API_KEY
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# USD per 1M tokens as (input, output); update if OpenAI changes its rates.
PRICES = {
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o":            (2.50, 10.00),
    "gpt-4o-mini":       (0.15, 0.60),
}
MAX_BODY_CHARS = 6000   # longer bodies are cut to limit cost


def load_env():
    """Load OPENAI_API_KEY from the .env file if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
    except Exception:
        pass


def sample_test(n: int, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Draw a balanced, reproducible sample of n test emails (original row in "_row")."""
    te = pd.read_csv(READ_DIR / "test.csv").reset_index(drop=True)
    te["_row"] = te.index
    per = n // 2
    parts = [te[te.label == lab].sample(n=min(per, (te.label == lab).sum()),
                                        random_state=seed) for lab in (C.LEGIT, C.PHISH)]
    s = pd.concat(parts).sort_values("_row")
    for c in ("text", "subject", "body", "sender", "urls", "sender_domain", "sender_tld"):
        if c in s.columns:
            s[c] = s[c].fillna("").astype(str)
    return s


def truncate(rec: dict) -> dict:
    """Return a copy of the email with its body cut to MAX_BODY_CHARS."""
    r = dict(rec)
    if len(r.get("body", "")) > MAX_BODY_CHARS:
        r["body"] = r["body"][:MAX_BODY_CHARS] + " …[truncated]"
    return r


def few_shot_examples(seed: int = C.RANDOM_SEED):
    """Pick two legitimate and two phishing few-shot examples from the training split."""
    tr = pd.read_csv(READ_DIR / "train.csv")
    for c in ("text", "subject", "body", "sender", "sender_domain", "sender_tld", "urls"):
        if c in tr.columns:
            tr[c] = tr[c].fillna("").astype(str)
    ex = []
    for lab, verdict in [(C.LEGIT, "legitimate"), (C.PHISH, "phishing")]:
        for _, row in tr[tr.label == lab].sample(n=2, random_state=seed).iterrows():
            v = {"classification": verdict, "confidence": 0.9,
                 "severity": "high" if lab == C.PHISH else "low",
                 "iocs": [], "recommended_action":
                 "escalate" if lab == C.PHISH else "auto-close",
                 "reasoning": "example"}
            ex.append((truncate(row.to_dict()), v))
    return ex


def call_llm(client, messages, model):
    """Send one chat request and return (parsed JSON verdict, usage); retries up to 4 times."""
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=L.TEMPERATURE,
                response_format={"type": "json_object"})
            return json.loads(resp.choices[0].message.content), resp.usage
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def run_strategy(client, model, strategy, sample, examples, prices):
    """Classify the sample with one strategy, using and updating the cache.

    Returns (preds, scores, cost_usd, mean_latency_s, in_tokens, out_tokens).
    """
    cache_file = CACHE_DIR / f"{model}__{strategy}.json".replace("/", "_")
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    preds, scores, in_tok, out_tok, lat = [], [], 0, 0, []

    for i, (_, row) in enumerate(sample.iterrows()):
        key = str(int(row["_row"]))
        rec = truncate(row.to_dict())
        if key in cache:
            data = cache[key]["data"]
            in_tok += cache[key].get("in", 0); out_tok += cache[key].get("out", 0)
        else:
            msgs = L.build_prompt(rec, strategy=strategy,
                                  examples=examples if strategy == "few_shot" else None)
            t0 = time.time()
            data, usage = call_llm(client, msgs, model)
            lat.append(time.time() - t0)
            u_in = getattr(usage, "prompt_tokens", 0); u_out = getattr(usage, "completion_tokens", 0)
            in_tok += u_in; out_tok += u_out
            cache[key] = {"data": data, "in": u_in, "out": u_out}
            # Save every 20 answers so an interrupted run loses little work.
            if (i + 1) % 20 == 0:
                cache_file.write_text(json.dumps(cache))
            print(f"\r  {strategy}: {i+1}/{len(sample)}", end="", flush=True)
        cls = str(data.get("classification", "legitimate"))
        pred = 1 if cls.lower().startswith("phish") else 0
        conf = float(data.get("confidence", 0.5) or 0.5)
        preds.append(pred)
        scores.append(conf if pred == 1 else 1 - conf)
    cache_file.write_text(json.dumps(cache))
    print()

    pin, pout = prices
    cost = in_tok / 1e6 * pin + out_tok / 1e6 * pout
    return (np.array(preds), np.array(scores), cost,
            (np.mean(lat) if lat else 0.0), in_tok, out_tok)


def metrics_row(name, y_true, y_pred, scores=None, extra=None):
    """Return one result-table row with the metrics of a model."""
    m = E.metrics(y_true, y_pred)
    row = {"model": name, "precision": m["precision"], "recall": m["recall"],
           "f1": m["f1"], "accuracy": m["accuracy"], "fpr": m["fpr"]}
    try:
        from sklearn.metrics import roc_auc_score
        row["roc_auc"] = roc_auc_score(y_true, scores) if scores is not None else float("nan")
    except Exception:
        row["roc_auc"] = float("nan")
    if extra:
        row.update(extra)
    return row


def main() -> int:
    """Run each strategy, add the baselines and save the table."""
    # --- 1. Read the options and connect to OpenAI ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--model", default=None)
    ap.add_argument("--strategies", nargs="+",
                    default=["zero_shot", "few_shot", "chain_of_thought"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    load_env()
    from openai import OpenAI
    client = OpenAI()

    if args.smoke:
        args.n, args.model, args.strategies = 6, "gpt-4o-mini", ["zero_shot"]
    model = args.model or L.DEFAULT_MODEL
    prices = PRICES.get(model, (0.0, 0.0))
    print(f"Model={model}  n={args.n}  strategies={args.strategies}")

    # --- 2. Draw the sample and the few-shot examples ---
    sample = sample_test(args.n)
    y = sample["label"].to_numpy(int)
    examples = few_shot_examples() if "few_shot" in args.strategies else None
    print(f"Sample: {len(sample)} emails ({(y==1).sum()} phishing / {(y==0).sum()} legit)\n")

    # --- 3. Run each prompting strategy ---
    rows, total_cost = [], 0.0
    for strat in args.strategies:
        preds, scores, cost, lat, itk, otk = run_strategy(
            client, model, strat, sample, examples, prices)
        total_cost += cost
        r = metrics_row(f"llm_{strat}", y, preds, scores,
                        {"avg_latency_s": round(lat, 2), "est_cost_usd": round(cost, 4),
                         "in_tok": itk, "out_tok": otk})
        rows.append(r)
        print(f"  -> P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"AUC={r['roc_auc']:.3f} FPR={r['fpr']:.3f}  cost=${cost:.3f}\n")

    # --- 4. Score the baselines on the same emails (if step 03 was run) ---
    bpath = SAVE_DIR / "uc1_predictions.csv"
    if bpath.exists():
        bp = pd.read_csv(bpath)
        idx = sample["_row"].to_numpy()
        for name in ("tfidf_lr", "tfidf_rf", "tfidf_lr_meta"):
            if f"{name}_pred" in bp.columns:
                rows.append(metrics_row(name + "_on_sample", y,
                            bp[f"{name}_pred"].to_numpy()[idx],
                            bp.get(f"{name}_score", pd.Series(np.zeros(len(bp)))).to_numpy()[idx]))

    # --- 5. Save and print the results ---
    out = pd.DataFrame(rows)
    tag = "smoke" if args.smoke else str(args.n)
    out.to_csv(SAVE_DIR / f"uc1_llm_{tag}.csv", index=False)
    cols = ["model", "precision", "recall", "f1", "accuracy", "roc_auc", "fpr"]
    print("================= UC1: LLM vs BASELINES (same sample) =================")
    print(out[cols].to_string(index=False))
    print(f"\nTotal estimated API cost this run: ${total_cost:.3f}")
    print(f"Saved -> {SAVE_DIR / f'uc1_llm_{tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
