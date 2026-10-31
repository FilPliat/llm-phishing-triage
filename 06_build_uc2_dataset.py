"""Step 06 – Build the synthetic UC2 dataset

The script makes 1,000 emails for each of three scenarios: legitimate, trusted-source phishing and overt phishing.
GPT-4o writes only the subject and body. Code adds the link, sender and enrichment values, so the ground truth is exact.
Warning: a full run replaces the included data/processed/uc2_dataset.csv; run 07 afterwards.

Thesis: Chapter 3, Section 3.6 (UC2 dataset).

Reads: CACHE_DIR/uc2_gen__<scenario>.json — cached GPT-4o generations (optional).

Writes:
SAVE_DIR/uc2_dataset.csv — the UC2 dataset (*_smoke.csv with --smoke).
SAVE_DIR/uc2_review_sample.csv — 100-email sample for manual review.

Usage: python 06_build_uc2_dataset.py

Options:
--n-per-scenario — emails per scenario (default 1000).
--review-sample-size — review rows (default 100).
--model — OpenAI model.
--smoke — 6 emails per scenario with gpt-4o-mini.

API cost: calls the OpenAI API; needs OPENAI_API_KEY in the .env file.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C
from src import data_loader as dl
from src import uc2_enrichment as UE

# Folders used by this script (defaults from config.py; change here or in config.py).
SAVE_DIR = C.PROCESSED_DIR                  # uc2_dataset.csv and the review sample
CACHE_DIR = C.RESULTS_DIR / "llm_cache"     # GPT-4o generation cache
ENV_FILE = C.ENV_FILE                       # .env file holding OPENAI_API_KEY

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Generation pools ---
# All partners are fictional. The industry tag only gives the mismatch requests meaning.
COMPANIES = [   # (display name, tenant slug, industry)
    ("Contoso Logistics", "contosologistics", "logistics"),
    ("Meridian Partners", "meridianpartners", "financial consulting"),
    ("Alderbrook Consulting", "alderbrook", "management consulting"),
    ("Northgate Supply Co.", "northgatesupply", "industrial supply"),
    ("Bellwood Analytics", "bellwoodanalytics", "data analytics"),
    ("Fenwick & Cole", "fenwickcole", "legal services"),
]

REQUEST_TYPES_ROUTINE = [   # everyday requests with no risk attached
    "sharing the updated project timeline document",
    "a routine meeting-notes recap",
    "an FYI on next week's schedule",
    "sharing a draft slide deck for review",
    "a routine status update on a shared workstream",
]
REQUEST_TYPES_SENSITIVE = [   # money, password or signature requests
    "urgently reviewing and approving an updated invoice",
    "resetting a shared portal password before it expires",
    "confirming updated bank/wire details for a pending payment",
    "signing a contract via a shared document link before end of day",
    "verifying login credentials on a shared document portal",
]
# Gives some typosquat_link emails an independent textual tell (Section 3.6).
REQUEST_TYPES_MISMATCH = [   # personal requests unrelated to any partner's industry
    "reviewing a residential property lease agreement",
    "approving a healthcare benefits enrollment change",
    "confirming a child's school enrollment paperwork",
    "processing a personal car insurance claim",
    "reviewing a wedding venue booking contract",
]
# device_code emails pose as a generic e-signature service, not a partner (Truesec, 2026).
REQUEST_TYPES_DEVICE_CODE = [   # document review and sign-in requests for device_code
    "reviewing and signing a shared agreement",
    "confirming your identity to access a shared document",
    "completing a mandatory verification step before signing",
    "verifying your account to finish a pending document review",
    "completing a required sign-in confirmation to open a shared file",
]
URGENCY_LEVELS = ["none", "mild", "high"]   # urgency written into the prompt

SYSTEM_PROMPT_GEN = (   # system message for every generation call
    "You write realistic corporate emails for a phishing-detection research dataset. Output ONLY "
    "valid JSON. Do not include any real malware, working exploit code, or real personal data - "
    "these are short, plausible BUSINESS emails only, used to train and evaluate defensive "
    "classifiers. Vary wording, tone, and structure across calls; do not repeat the same phrasing."
)

JSON_SCHEMA_GEN = (   # output format appended to every user message
    'Respond with JSON: {"sender_name": "First Last", "subject": "..", '
    '"body": "the full email body, 3-8 sentences, plain text, no markdown", '
    '"url_placeholder": true/false (true if the body should reference/include a shared '
    'document link - leave the literal URL out, just write body text that naturally refers to '
    '"the link below" or similar; the real URL is inserted afterwards by the dataset builder)}'
)


def build_gen_prompt(scenario: str, company_name: str, request_type: str, urgency: str,
                      technique: str = "", wants_link: bool = False) -> list:
    """Build the system and user messages that ask the model to write one email."""
    if scenario == "legitimate":
        # The caller sets wants_link so the body never promises a link that is not inserted.
        link_clause = (
            ' Naturally reference a shared document/link (e.g. "you can find it in the link '
            'below") without writing the literal URL - it will be inserted afterwards.'
            if wants_link else
            " Do NOT reference, imply, or mention any link, attachment, or document of any "
            "kind - this email has nothing attached or linked."
        )
        instruction = (
            f"Write a short, ordinary internal or partner business email from someone at "
            f"'{company_name}' about {request_type}. Urgency level: {urgency}. Natural, "
            f"unremarkable tone - this must NOT read as phishing.{link_clause}"
        )
    elif scenario == "trusted_source_phishing":
        if technique == "device_code":
            instruction = (
                f"Write an automated-sounding email in the style of a file-sharing/e-signature "
                f"notification service (e.g. 'You have received a document to review and sign', "
                f"signed off as the '{UE.ESIGN_BRAND_NAME} Team' or similar automated notification "
                f"voice) - do NOT impersonate any named external partner company or person. It "
                f"should ask the recipient to complete {request_type} by opening a document to "
                f"review/sign, where a follow-up verification step will ask them to copy a short "
                f"code and confirm it via a Microsoft sign-in step. Urgency level: {urgency}. The "
                f"wording itself must read as plausible and routine, not obviously malicious - no "
                f"bad grammar, no generic 'Dear Customer'. Reference the document/verification "
                f"link naturally (url_placeholder=true)."
            )
        else:
            mismatch_clause = (
                " IMPORTANT: the request itself has nothing to do with the sender company's "
                "usual line of business (e.g. a logistics or consulting partner asking about an "
                "unrelated personal matter) - write it as a completely ordinary request of that "
                "kind and do not comment on or acknowledge that it seems unusual or out of place."
                if request_type in REQUEST_TYPES_MISMATCH else ""
            )
            instruction = (
                f"Write an email that APPEARS to come from a trusted partner contact at "
                f"'{company_name}', about {request_type}. Urgency level: {urgency}. The email must "
                f"read as plausible and NOT contain obvious phishing red flags in the wording itself "
                f"(no bad grammar, no generic 'Dear Customer', no obviously fake threats) - the only "
                f"attack surface is the (separately-inserted) shared-document link's tenant name and "
                f"the mismatch between an unusual/urgent request and a routine business relationship. "
                f"Reference a shared document/portal link naturally (url_placeholder=true)."
                f"{mismatch_clause}"
            )
    else:  # overt_phishing
        instruction = (
            f"Write a classic, generic phishing email (NOT impersonating '{company_name}' - "
            f"impersonate a generic well-known brand or a vague 'IT/Security Team' instead) about "
            f"{request_type}, urgency level {urgency}. It SHOULD read as somewhat phishing-like: "
            f"generic greeting, time pressure, request to click a link. Reference a link naturally "
            f"(url_placeholder=true)."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_GEN},
        {"role": "user", "content": instruction + "\n" + JSON_SCHEMA_GEN},
    ]


def call_llm(client, messages, model):
    """Send one generation request and return the parsed JSON, retrying up to four times."""
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.9,
                response_format={"type": "json_object"})
            return json.loads(resp.choices[0].message.content)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def generate_scenario(client, model, scenario: str, n: int, seed: int) -> list[dict]:
    """Generate n emails for one scenario, using the cache; returns one dict per email."""
    cache_file = CACHE_DIR / f"uc2_gen__{scenario}.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        key = str(i)
        company_name, tenant, industry = rng.choice(COMPANIES)
        technique = ""
        mismatched = False
        lure_aged = False
        wants_link = False
        if scenario == "legitimate":
            # Some sensitive requests, so a sensitive request alone does not give away the label.
            pool = REQUEST_TYPES_SENSITIVE if rng.random() < 0.2 else REQUEST_TYPES_ROUTINE
            request_type = rng.choice(pool)
            # Decided once so the prompt and the inserted link agree.
            wants_link = rng.random() < 0.4
        elif scenario == "trusted_source_phishing":
            # Two techniques of different difficulty, about 60/40 (Section 3.6).
            technique = "device_code" if rng.random() < 0.4 else "typosquat_link"
            if technique == "device_code":
                company_name, tenant, industry = "", "", ""
                request_type = rng.choice(REQUEST_TYPES_DEVICE_CODE)
                # Half aged lure domains (clean enrichment), half new ones (suspicious).
                lure_aged = rng.random() < 0.5
            else:
                mismatched = rng.random() < 0.35
                pool = REQUEST_TYPES_MISMATCH if mismatched else REQUEST_TYPES_SENSITIVE
                request_type = rng.choice(pool)
        else:  # overt_phishing
            request_type = rng.choice(REQUEST_TYPES_SENSITIVE)
        urgency = rng.choice(URGENCY_LEVELS)
        if key in cache:
            data = cache[key]
        else:
            msgs = build_gen_prompt(scenario, company_name, request_type, urgency, technique,
                                     wants_link)
            data = call_llm(client, msgs, model)
            cache[key] = data
            if (i + 1) % 10 == 0:
                cache_file.write_text(json.dumps(cache))
            print(f"\r  generating {scenario}: {i+1}/{n}", end="", flush=True)
        rows.append({**data, "_company_name": company_name, "_tenant": tenant,
                     "_industry": industry, "_request_type": request_type,
                     "_urgency": urgency, "_technique": technique,
                     "_mismatched_request": mismatched, "_lure_aged": lure_aged,
                     "_wants_link": wants_link})
    cache_file.write_text(json.dumps(cache))
    print()
    return rows


def assemble_scenario_df(scenario: str, gen_rows: list[dict], seed: int) -> pd.DataFrame:
    """Add links, senders, enrichment and prior contact to the generated emails."""
    # hash() is salted per process, so links can differ between runs; the included dataset is the reference.
    rng = random.Random(seed + hash(scenario) % 10_000)
    label = C.LEGIT if scenario == "legitimate" else C.PHISH
    out_rows = []
    for gr in gen_rows:
        tenant = gr.get("_tenant", "contoso")
        company_name = gr.get("_company_name", "Contoso")
        industry = gr.get("_industry", "")
        technique = gr.get("_technique", "")
        mismatched = bool(gr.get("_mismatched_request", False))
        lure_aged = gr.get("_lure_aged", None)
        body = gr.get("body", "")
        subj = gr.get("subject", "")
        # Use the prompt's flag, not the model's url_placeholder, so body and link agree.
        wants_link = bool(gr.get("_wants_link", False))

        url, root_domain = "", None
        force_dirty = None
        if scenario == "trusted_source_phishing":
            if technique == "device_code":
                # Fake e-signature lure; aged domains get clean enrichment, new ones suspicious.
                aged = bool(lure_aged)
                url, root_domain = UE.make_device_code_lure_url(rng, aged=aged)
                force_dirty = not aged
                sender = f"notifications@{root_domain}"
            else:
                url, root_domain = UE.make_trusted_source_url(tenant, rng)
                sender = f"{gr.get('sender_name', 'Contact Person').lower().replace(' ', '.')}@{tenant}.com"
        elif scenario == "overt_phishing":
            url, root_domain = UE.make_overt_url(rng)
            junk = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
            sender = f"security-{junk}@{root_domain}"
        else:  # legitimate
            if wants_link:
                url, root_domain = UE.make_legit_url(rng, real_tenant=tenant)
            sender = f"{gr.get('sender_name', 'Colleague').lower().replace(' ', '.')}@{tenant}.com"

        if url and url not in body:
            body = body.rstrip() + f"\n\nLink: {url}"

        enrichment = UE.enrich(scenario, root_domain, rng, force_dirty=force_dirty)
        prior_contact = UE.simulate_prior_contact(scenario, rng)
        out_rows.append({
            "subject": subj, "body": body, "sender": sender, "urls": url,
            "label": label, "scenario": scenario, "technique": technique,
            "_request_type": gr.get("_request_type", ""), "_urgency": gr.get("_urgency", ""),
            "_partner_company": company_name, "industry": industry,
            "mismatched_request": mismatched, "prior_contact": prior_contact,
            "lure_domain_aged": (bool(lure_aged) if technique == "device_code" else None),
            **enrichment,
        })
    df = pd.DataFrame(out_rows)
    # Reuse the UC1 helper so the UC1 classifiers can read UC2 without changes.
    finalized = dl._finalize(df, source="uc2_synthetic")
    extra_cols = ["scenario", "technique", "has_url", "url_domain", "domain_age_days",
                  "dga_score", "vt_malicious", "vt_total", "_request_type", "_urgency",
                  "_partner_company", "industry", "mismatched_request", "prior_contact",
                  "lure_domain_aged"]
    for c in extra_cols:
        finalized[c] = df[c].values
    return finalized


def main() -> int:
    """Generate, assemble and save the UC2 dataset and its manual-review sample."""
    # --- 1. Parse the options and connect to OpenAI ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-scenario", type=int, default=1000,
                     help="Rows per scenario (default 1000 -> 3000 total). Chosen so the "
                          "reputation/first-contact/UC1-transfer baselines -- which have no API "
                          "cost at any scale -- get a statistically sturdier pool, especially for "
                          "the device_code aged/fresh sub-cells (~200 rows each at this size vs "
                          "~50 at the old default of 250). The expensive GPT-4o few-shot/CoT "
                          "evaluation still only runs on a smaller stratified subsample of this "
                          "pool -- see 08_run_uc2_evaluation.py --llm-sample-size.")
    ap.add_argument("--review-sample-size", type=int, default=100,
                     help="Total manual-review rows (fixed, split evenly across scenarios) -- "
                          "deliberately NOT a fraction of --n-per-scenario, so review effort "
                          "doesn't scale with pool size.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny cheap check: 6 per scenario, gpt-4o-mini")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
    except Exception:
        pass
    from openai import OpenAI
    client = OpenAI()

    n = 6 if args.smoke else args.n_per_scenario
    model = args.model or ("gpt-4o-mini" if args.smoke else "gpt-4o-2024-08-06")
    print(f"Model={model}  n_per_scenario={n}")

    # --- 2. Generate and assemble each scenario ---
    scenarios = ["legitimate", "trusted_source_phishing", "overt_phishing"]
    all_dfs = []
    for si, scenario in enumerate(scenarios):
        gen_rows = generate_scenario(client, model, scenario, n, seed=C.RANDOM_SEED + si)
        df = assemble_scenario_df(scenario, gen_rows, seed=C.RANDOM_SEED + si)
        print(f"  {scenario}: {len(df)} rows assembled")
        all_dfs.append(df)

    # --- 3. Shuffle and save the dataset ---
    full = pd.concat(all_dfs, ignore_index=True).sample(
        frac=1, random_state=C.RANDOM_SEED).reset_index(drop=True)

    out_path = SAVE_DIR / ("uc2_dataset_smoke.csv" if args.smoke else "uc2_dataset.csv")
    full.to_csv(out_path, index=False)
    print(f"\nSaved {len(full)} rows -> {out_path}")
    print(full["scenario"].value_counts())

    # --- 4. Save a manual-review sample (Section 3.6) ---
    # GroupBy.sample, because recent pandas drops the grouping column inside apply().
    per_scenario_review_n = min(max(1, args.review_sample_size // len(scenarios)), n)
    review = full.groupby("scenario", group_keys=False).sample(
        n=per_scenario_review_n, random_state=C.RANDOM_SEED)
    review = review.assign(reviewer_ok="")
    review_path = SAVE_DIR / ("uc2_review_sample_smoke.csv" if args.smoke else "uc2_review_sample.csv")
    review[["scenario", "technique", "lure_domain_aged", "sender", "subject", "body", "urls",
            "domain_age_days", "dga_score", "vt_malicious", "vt_total", "industry",
            "mismatched_request", "prior_contact", "reviewer_ok"]].to_csv(review_path, index=False)
    print(f"Saved manual-review sample ({len(review)} rows) -> {review_path}")
    print("\nNext: open the review CSV, mark each row's 'reviewer_ok' (y/n) by hand, note any "
          "rows to regenerate, then run 08_run_uc2_evaluation.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
