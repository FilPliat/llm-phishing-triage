"""GPT-4o triage classifier: email formatting, prompt building and the JSON verdict.

Temperature 0 and a pinned model snapshot keep the results as reproducible as the API allows.

Thesis: Sections 3.2-3.3 (triage tool and prompt design).
Used by: 04_run_uc1_llm.py, 08_run_uc2_evaluation.py, 09_explain_uc2_shap.py.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-2024-08-06")   # pinned, dated model snapshot
TEMPERATURE = 0.0                                              # 0 = most deterministic answers


@dataclass
class TriageVerdict:
    """The structured triage verdict the model returns for one email."""

    classification: str
    confidence: float
    severity: str
    iocs: list
    recommended_action: str
    reasoning: str

    def label(self) -> int:
        """Return 1 if the classification starts with "phish", otherwise 0."""
        return 1 if self.classification.lower().startswith("phish") else 0


SYSTEM_PROMPT = (
    "You are a SOC analyst. You assess whether an email is phishing and produce a structured triage verdict. Consider both the message content AND the header/link signals provided (sender address and domain, top-level domain, and URLs). A body that impersonates a trusted brand while the sender domain or links do not match that brand is a strong phishing signal. In a SOC, false positives are costly, so only flag phishing when the evidence supports it. Return ONLY valid JSON matching the requested schema."
)

JSON_INSTRUCTION = (
    'Respond with JSON: {"classification": "phishing|legitimate", '
    '"confidence": 0.0-1.0, "severity": "low|medium|high|critical", '
    '"iocs": [..], "recommended_action": "..", "reasoning": ".."}'
)

# The first prompt judged links by root-domain reputation and often did not compare the
# tenant name with the sender's organization. Applied only to UC2 (the "improved" prompt).
SYSTEM_PROMPT_UC2_TENANT_ADDENDUM = (
    "\n\nAdditional guidance for shared-document/e-signature links: a link's ROOT domain being reputable, old and clean on VirusTotal/blacklist hits (e.g. sharepoint.com, onedrive.live.com, or any well-known cloud platform) does NOT by itself mean the email is safe. Attackers host phishing content on such platforms using a stolen or purchased account. You must separately check the TENANT NAME or organization identifier embedded in the link (e.g. the subdomain segment before 'my.sharepoint.com', or a tenant/authkey parameter in a onedrive.live.com URL) against the sender's own claimed organization. Even a single added hyphen, swapped letter or doubled character in that tenant name is a phishing indicator on its own regardless of how old or clean the root domain is. Also, if the email signs off as, or claims to represent a specific named service or brand, verify that the sender's actual domain belongs to that claimed brand. An unrelated domain claiming to be a specific named brand is suspicious even if that domain is old and has no blacklist hits. 'The sender domain matches the link domain' is NOT sufficient evidence of legitimacy if neither domain actually matches the identity or organization the email content claims to be."
)


def format_email(record) -> str:
    """Render one email (dict, Series or plain body string) as the text block the model reads.

    The enrichment line appears only for UC2 records, which have the enrichment fields.
    """
    if isinstance(record, str):
        return f"Body:\n\"\"\"\n{record}\n\"\"\""
    g = record.get
    sender = g("sender", "") or ""
    dom = g("sender_domain", "") or ""
    tld = g("sender_tld", "") or ""
    urls = g("urls", "") or ""
    subj = g("subject", "") or ""
    body = g("body", "") or g("text", "") or ""
    lines = []
    if sender or dom:
        tag = f"  (domain: {dom}, TLD: .{tld})" if dom else ""
        lines.append(f"From: {sender}{tag}")
    if subj:
        lines.append(f"Subject: {subj}")
    if urls:
        lines.append(f"URLs: {urls}")

    has_url = g("has_url", None)
    if has_url is not None:
        if has_url:
            age = g("domain_age_days", None)
            dga = g("dga_score", None)
            vt_m, vt_t = g("vt_malicious", None), g("vt_total", None)
            parts = []
            if age is not None:
                parts.append(f"link domain age {age} days")
            if dga is not None:
                parts.append(f"DGA-likelihood score {dga}")
            if vt_m is not None and vt_t is not None:
                parts.append(f"VirusTotal {vt_m}/{vt_t} engines flag it malicious")
            if parts:
                lines.append("Automated enrichment: " + "; ".join(parts) + ".")
        else:
            lines.append("Automated enrichment: no URL present to enrich.")

    lines.append(f"Body:\n\"\"\"\n{body}\n\"\"\"")
    return "\n".join(lines)


def build_prompt(record, strategy: str = "zero_shot",
                 examples: list | None = None, context: str | None = None,
                 system_addendum: str | None = None) -> list:
    """Build the chat messages that ask the model to triage one email.

    Args:
        strategy: "zero_shot", "few_shot" or "chain_of_thought".
        examples: Few-shot only; (email record, verdict dict) pairs.
        context: Optional note on the sender relationship.
        system_addendum: Text appended to SYSTEM_PROMPT (e.g. the UC2 tenant addendum).
    """
    user = []
    if strategy == "chain_of_thought":
        user.append("Reason step by step about the sender domain, the request, urgency and the links. Then give your verdict.")
    if context:
        user.append(f"Context on the sender relationship:\n{context}")
    user.append(format_email(record))
    user.append(JSON_INSTRUCTION)

    system_content = SYSTEM_PROMPT + (system_addendum or "")
    messages = [{"role": "system", "content": system_content}]
    if strategy == "few_shot" and examples:
        for ex_record, ex_verdict in examples:
            messages.append({"role": "user", "content": format_email(ex_record) + "\n" + JSON_INSTRUCTION})
            messages.append({"role": "assistant", "content": json.dumps(ex_verdict)})
    messages.append({"role": "user", "content": "\n\n".join(user)})
    return messages


def classify(record, strategy: str = "zero_shot", model: str = DEFAULT_MODEL, **kwargs) -> TriageVerdict:
    """Send one email to the OpenAI API and parse the answer into a TriageVerdict.

    Extra keyword arguments go to build_prompt(). Missing answer fields get safe defaults.
    """
    from openai import OpenAI
    client = OpenAI()
    messages = build_prompt(record, strategy=strategy, **kwargs)
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=TEMPERATURE,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return TriageVerdict(
        classification=data.get("classification", "legitimate"),
        confidence=float(data.get("confidence", 0.0)),
        severity=data.get("severity", "low"),
        iocs=data.get("iocs", []),
        recommended_action=data.get("recommended_action", ""),
        reasoning=data.get("reasoning", ""),
    )

# Batch runs, caching and cost logging live in 04_run_uc1_llm.py and 08_run_uc2_evaluation.py.
