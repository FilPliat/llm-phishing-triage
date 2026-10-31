"""Step 08 – Evaluate all seven approaches on UC2

The script runs the reputation baseline, the first-contact heuristic, the transferred UC1 model and GPT-4o (few-shot and chain-of-thought, each with the baseline and improved prompt).
It reports results overall, per scenario and per trusted-source technique.
The non-LLM approaches use all 3,000 emails; GPT-4o uses a 750-email stratified subsample, so the "n" column differs.

Thesis: Sections 3.3–3.4; results in Section 4.2 (Tables 4–6, Figures 3–4).

Reads:
DATASET_FILE — the UC2 dataset from 06 and 07
UC1_READ_DIR/train.csv — UC1 training split from 02_prepare_data.py

Writes:
SAVE_DIR/uc2_overall.csv, uc2_per_scenario.csv, uc2_per_technique.csv — result tables.
SAVE_DIR/uc2_predictions.csv — every approach's prediction for every email.
CACHE_DIR/uc2_eval__*.json — cached GPT-4o answers.

Usage: python 08_run_uc2_evaluation.py

Options:
--skip-llm — skip GPT-4o.
--prompt-variants — baseline,improved (default both).
--llm-sample-size — GPT-4o sample size (default 750).
--llm-sample-seed — seed for that sample.

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
from src import baselines as B
from src import evaluation as E
from src import llm_classifier as L

# Folders used by this script (defaults from config.py; change here or in config.py).
UC1_READ_DIR = C.PROCESSED_DIR                      # UC1 train.csv (from 02)
DATASET_FILE = C.PROCESSED_DIR / "uc2_dataset.csv"  # UC2 email pool (from 06)
SAVE_DIR = C.RESULTS_DIR                            # UC2 result tables
CACHE_DIR = C.RESULTS_DIR / "llm_cache"             # cached GPT-4o answers
ENV_FILE = C.ENV_FILE                               # .env file holding OPENAI_API_KEY

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# USD per 1M (input, output) tokens, for the cost estimate.
PRICES = {"gpt-4o-2024-08-06": (2.50, 10.00), "gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}


def load_uc2() -> pd.DataFrame:
    """Load the UC2 email pool and clean the columns the approaches need."""
    path = DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run 06_build_uc2_dataset.py first.")
    df = pd.read_csv(path)
    for col in ("text", "sender_tld", "urls", "scenario"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def per_scenario_table(approach: str, df: pd.DataFrame, y_pred: np.ndarray) -> list[dict]:
    """Return the correct rate per scenario (specificity for legitimate, recall for phishing)."""
    rows = []
    for scenario, grp_idx in df.groupby("scenario").groups.items():
        idx = np.asarray(grp_idx)
        y_true_s = df.loc[idx, "label"].to_numpy(int)
        y_pred_s = y_pred[idx]
        correct = (y_true_s == y_pred_s).mean()
        rows.append({"approach": approach, "scenario": scenario, "n": len(idx),
                     "correct_rate": round(float(correct), 4)})
    return rows


def per_technique_table(approach: str, df: pd.DataFrame, y_pred: np.ndarray) -> list[dict]:
    """Return trusted-source phishing recall for each attack technique (Table 6)."""
    rows = []
    mask = (df["scenario"] == "trusted_source_phishing") & (df["technique"].fillna("") != "")
    sub = df.loc[mask]
    for technique, grp_idx in sub.groupby("technique").groups.items():
        idx = np.asarray(grp_idx)
        y_pred_s = y_pred[idx]
        recall = float((y_pred_s == 1).mean()) if len(idx) else float("nan")
        rows.append({"approach": approach, "technique": technique, "n": len(idx),
                     "recall": round(recall, 4)})
    return rows


def stratified_llm_sample(df: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    """Draw the scenario-stratified GPT-4o subsample, keeping the original index."""
    scenarios = df["scenario"].unique()
    smallest_group = df.groupby("scenario").size().min()
    per_scenario_n = max(1, n_target // len(scenarios))
    if per_scenario_n >= smallest_group:
        return df
    return df.groupby("scenario", group_keys=False).sample(n=per_scenario_n, random_state=seed)


def run_reputation(df: pd.DataFrame) -> np.ndarray:
    """Run the reputation baseline (1 = phishing)."""
    from src import reputation_baseline as R
    return R.predict(df)


def run_first_contact(df: pd.DataFrame) -> np.ndarray:
    """Run the first-contact heuristic (1 = phishing)."""
    from src import reputation_baseline as R
    return R.predict_first_contact(df)


def run_uc1_transfer(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Train tfidf_lr_meta on UC1 and apply it to UC2; returns predictions and fit time."""
    train = pd.read_csv(UC1_READ_DIR / "train.csv")
    for col in ("text", "sender_tld"):
        train[col] = train[col].fillna("").astype(str)
    train["num_urls"] = pd.to_numeric(train["num_urls"], errors="coerce").fillna(0)
    df = df.copy()
    df["num_urls"] = pd.to_numeric(df["num_urls"], errors="coerce").fillna(0)

    builder, _ = B.MODELS["tfidf_lr_meta"]
    model = builder()
    t0 = time.time()
    model.fit(train[["text", "sender_tld", "num_urls"]], train["label"].to_numpy(int))
    fit_s = time.time() - t0
    preds = model.predict(df[["text", "sender_tld", "num_urls"]]).astype(int)
    return preds, fit_s


def run_llm(df: pd.DataFrame, strategy: str, model_name: str,
            system_addendum: str | None = None, cache_tag: str = "") -> tuple[np.ndarray, np.ndarray, float]:
    """Classify every email with GPT-4o (cached); returns predictions, scores and cost."""
    from openai import OpenAI
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
    except Exception:
        pass
    client = OpenAI()

    examples = None
    if strategy == "few_shot":
        # Same few-shot examples as 04, so the UC1 and UC2 prompts are built alike.
        tr = pd.read_csv(UC1_READ_DIR / "train.csv")
        for c in ("text", "subject", "body", "sender", "sender_domain", "sender_tld", "urls"):
            if c in tr.columns:
                tr[c] = tr[c].fillna("").astype(str)
        examples = []
        for lab, verdict in [(C.LEGIT, "legitimate"), (C.PHISH, "phishing")]:
            for _, row in tr[tr.label == lab].sample(n=2, random_state=C.RANDOM_SEED).iterrows():
                v = {"classification": verdict, "confidence": 0.9,
                     "severity": "high" if lab == C.PHISH else "low", "iocs": [],
                     "recommended_action": "escalate" if lab == C.PHISH else "auto-close",
                     "reasoning": "example"}
                examples.append((row.to_dict(), v))

    cache_file = CACHE_DIR / f"uc2_eval__{model_name}__{strategy}{cache_tag}.json".replace("/", "_")
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    preds, scores, in_tok, out_tok = [], [], 0, 0

    for i, (_, row) in enumerate(df.iterrows()):
        key = str(i)
        rec = row.to_dict()
        if key in cache:
            data = cache[key]["data"]
            in_tok += cache[key].get("in", 0); out_tok += cache[key].get("out", 0)
        else:
            msgs = L.build_prompt(rec, strategy=strategy, examples=examples if strategy == "few_shot" else None,
                                   system_addendum=system_addendum)
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(
                        model=model_name, messages=msgs, temperature=L.TEMPERATURE,
                        response_format={"type": "json_object"})
                    data = json.loads(resp.choices[0].message.content)
                    u_in = getattr(resp.usage, "prompt_tokens", 0)
                    u_out = getattr(resp.usage, "completion_tokens", 0)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            in_tok += u_in; out_tok += u_out
            cache[key] = {"data": data, "in": u_in, "out": u_out}
            if (i + 1) % 20 == 0:
                cache_file.write_text(json.dumps(cache))
            print(f"\r  llm_{strategy}: {i+1}/{len(df)}", end="", flush=True)
        cls = str(data.get("classification", "legitimate"))
        pred = 1 if cls.lower().startswith("phish") else 0
        conf = float(data.get("confidence", 0.5) or 0.5)
        preds.append(pred)
        scores.append(conf if pred == 1 else 1 - conf)
    cache_file.write_text(json.dumps(cache))
    print()

    pin, pout = PRICES.get(model_name, (0.0, 0.0))
    cost = in_tok / 1e6 * pin + out_tok / 1e6 * pout
    return np.array(preds), np.array(scores), cost


def main() -> int:
    """Run all UC2 approaches, print the result tables and save them as CSV files."""
    # --- 1. Parse the options and load the data ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-sample-size", type=int, default=750,
                     help="Total rows (stratified by scenario) sent to the expensive GPT-4o "
                          "few-shot/chain-of-thought evaluation. The three non-LLM approaches "
                          "always run on the FULL loaded pool regardless of this setting.")
    ap.add_argument("--llm-sample-seed", type=int, default=None,
                     help="Defaults to config.RANDOM_SEED if not given.")
    ap.add_argument("--skip-llm", action="store_true",
                     help="Run only the three non-LLM approaches -- no OpenAI calls. "
                          "Useful for testing the full-pool plumbing before spending anything.")
    ap.add_argument("--prompt-variants", default="baseline,improved",
                     help="Comma-separated subset of {baseline, improved} to run for the two "
                          "GPT-4o strategies (prompt ablation, thesis Section 3.3). 'baseline' is the original "
                          "SYSTEM_PROMPT -- already fully cached from the first full run, so "
                          "re-running it costs nothing. 'improved' adds "
                          "llm_classifier.SYSTEM_PROMPT_UC2_TENANT_ADDENDUM, written after "
                          "inspecting baseline miss reasoning showed the model checking link "
                          "root-domain reputation but never the tenant-name string embedded in "
                          "the link -- see that addendum's docstring for the full diagnosis. "
                          "Each variant gets its own cache file and approach name "
                          "(llm_<strategy> vs llm_<strategy>_improved_prompt) so both are "
                          "reported side by side rather than one overwriting the other.")
    args = ap.parse_args()
    llm_seed = args.llm_sample_seed if args.llm_sample_seed is not None else C.RANDOM_SEED

    df = load_uc2().reset_index(drop=True)
    y = df["label"].to_numpy(int)
    print(f"UC2 dataset: {len(df)} emails\n{df['scenario'].value_counts()}\n")

    overall_rows, per_scenario_rows, per_technique_rows = [], [], []
    preds_out = {"scenario": df["scenario"], "technique": df.get("technique", ""), "label": y}

    # --- 2. Run the three non-LLM approaches on the full pool ---
    print("=== reputation_baseline (full pool) ===")
    rep_pred = run_reputation(df)
    m = E.report("reputation_baseline", y, rep_pred)
    overall_rows.append({"approach": "reputation_baseline", "n": len(df), **m})
    per_scenario_rows += per_scenario_table("reputation_baseline", df, rep_pred)
    per_technique_rows += per_technique_table("reputation_baseline", df, rep_pred)
    preds_out["reputation_baseline_pred"] = rep_pred

    print("\n=== first_contact_heuristic (full pool) ===")
    fc_pred = run_first_contact(df)
    m = E.report("first_contact_heuristic", y, fc_pred)
    overall_rows.append({"approach": "first_contact_heuristic", "n": len(df), **m})
    per_scenario_rows += per_scenario_table("first_contact_heuristic", df, fc_pred)
    per_technique_rows += per_technique_table("first_contact_heuristic", df, fc_pred)
    preds_out["first_contact_heuristic_pred"] = fc_pred

    print("\n=== uc1_tfidf_lr_meta (transferred, full pool, no retraining on UC2) ===")
    uc1_pred, fit_s = run_uc1_transfer(df)
    m = E.report("uc1_tfidf_lr_meta", y, uc1_pred)
    overall_rows.append({"approach": "uc1_tfidf_lr_meta", "n": len(df),
                          "fit_seconds": round(fit_s, 1), **m})
    per_scenario_rows += per_scenario_table("uc1_tfidf_lr_meta", df, uc1_pred)
    per_technique_rows += per_technique_table("uc1_tfidf_lr_meta", df, uc1_pred)
    preds_out["uc1_tfidf_lr_meta_pred"] = uc1_pred

    # --- 3. Run GPT-4o on the stratified subsample, once per prompt variant ---
    total_cost = 0.0
    if args.skip_llm:
        print("\n--skip-llm set: skipping GPT-4o few-shot/chain-of-thought (no API cost incurred).")
    else:
        llm_df = stratified_llm_sample(df, args.llm_sample_size, llm_seed)
        llm_positions = llm_df.index.to_numpy()          # positions in the full pool
        llm_df_seq = llm_df.reset_index(drop=True)        # 0..n-1 index for run_llm
        y_llm = llm_df_seq["label"].to_numpy(int)
        print(f"\nLLM evaluation subsample: {len(llm_df_seq)} of {len(df)} emails -- "
              f"{llm_df_seq['scenario'].value_counts().to_dict()}")

        preds_out["in_llm_sample"] = np.zeros(len(df), dtype=bool)
        preds_out["in_llm_sample"][llm_positions] = True

        PROMPT_VARIANTS = {
            # Variant name -> (system_addendum, cache_tag, approach_name_suffix).
            "baseline": (None, "", ""),
            "improved": (L.SYSTEM_PROMPT_UC2_TENANT_ADDENDUM, "__tenant_addendum", "_improved_prompt"),
        }
        model_name = L.DEFAULT_MODEL
        for variant in [v.strip() for v in args.prompt_variants.split(",") if v.strip()]:
            if variant not in PROMPT_VARIANTS:
                print(f"  (skipping unknown --prompt-variants entry '{variant}')")
                continue
            addendum, cache_tag, name_suffix = PROMPT_VARIANTS[variant]
            for strategy in ("few_shot", "chain_of_thought"):
                approach = f"llm_{strategy}{name_suffix}"
                print(f"\n=== {approach} ({model_name}) -- subsample only, prompt variant={variant} ===")
                llm_pred, llm_scores, cost = run_llm(llm_df_seq, strategy, model_name,
                                                      system_addendum=addendum, cache_tag=cache_tag)
                total_cost += cost
                m = E.report(approach, y_llm, llm_pred)
                try:
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(y_llm, llm_scores)
                except Exception:
                    auc = float("nan")
                overall_rows.append({"approach": approach, "n": len(llm_df_seq),
                                      "roc_auc": auc, "est_cost_usd": round(cost, 4), **m})
                per_scenario_rows += per_scenario_table(approach, llm_df_seq, llm_pred)
                per_technique_rows += per_technique_table(approach, llm_df_seq, llm_pred)

                # Emails outside the subsample get NaN in the full-length prediction columns.
                full_pred = np.full(len(df), np.nan)
                full_pred[llm_positions] = llm_pred
                full_score = np.full(len(df), np.nan)
                full_score[llm_positions] = llm_scores
                preds_out[f"{approach}_pred"] = full_pred
                preds_out[f"{approach}_score"] = np.round(full_score, 4)
                print(f"  cost so far: ${total_cost:.3f}")

    # --- 4. Save and print the result tables ---
    overall = pd.DataFrame(overall_rows)
    per_scenario = pd.DataFrame(per_scenario_rows)
    per_technique = pd.DataFrame(per_technique_rows)
    predictions = pd.DataFrame(preds_out)

    overall.to_csv(SAVE_DIR / "uc2_overall.csv", index=False)
    per_scenario.to_csv(SAVE_DIR / "uc2_per_scenario.csv", index=False)
    per_technique.to_csv(SAVE_DIR / "uc2_per_technique.csv", index=False)
    predictions.to_csv(SAVE_DIR / "uc2_predictions.csv", index=False)

    print("\n================= UC2 OVERALL =================")
    cols = [c for c in ["approach", "precision", "recall", "f1", "accuracy", "roc_auc", "fpr"]
            if c in overall.columns]
    print(overall[cols].to_string(index=False))

    print("\n================= UC2 PER SCENARIO (correct_rate) =================")
    pivot = per_scenario.pivot(index="approach", columns="scenario", values="correct_rate")
    print(pivot.to_string())

    if not per_technique.empty:
        print("\n========= UC2 PER TECHNIQUE, trusted_source_phishing only (recall) =========")
        pivot_t = per_technique.pivot(index="approach", columns="technique", values="recall")
        print(pivot_t.to_string())

    print(f"\nTotal estimated LLM API cost this run: ${total_cost:.3f}")
    print(f"Saved -> {SAVE_DIR / 'uc2_overall.csv'}, uc2_per_scenario.csv, "
          f"uc2_per_technique.csv, uc2_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
