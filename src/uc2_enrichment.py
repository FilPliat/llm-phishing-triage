"""Simulated link enrichment, links and prior contact for the UC2 synthetic dataset.

No live lookups are made; value ranges were calibrated against a few manual lookups. Typosquatted
tenant links sit on genuine Microsoft roots, so their enrichment is as clean as legitimate mail.

Thesis: Sections 3.6-3.7 (UC2 dataset, scope and threat model).
Used by: 06_build_uc2_dataset.py.
"""
from __future__ import annotations
import random
import re
import string

# --- Calibration constants ---
OLD_DOMAIN_AGE_RANGE = (1500, 4500)     # days; sharepoint.com or an established brand domain
NEW_DOMAIN_AGE_RANGE = (1, 60)          # days; a freshly registered phishing domain
CLEAN_DGA_RANGE = (0.0, 0.15)           # DGA score of an ordinary, readable domain
DIRTY_DGA_RANGE = (0.55, 0.95)          # DGA score of a random-looking phishing domain
VT_TOTAL_RANGE = (60, 75)               # typical number of VirusTotal engines that scan a URL
DIRTY_VT_HITS_RANGE = (3, 16)           # engines flagging a suspicious domain

TRUSTED_URL_ROOTS = ["sharepoint.com", "onedrive.live.com"]   # genuine Microsoft roots
SUSPICIOUS_TLDS = ["xyz", "top", "info", "click", "live", "cfd", "gq"]   # cheap phishing TLDs
# Words that phishing domains commonly use to sound official.
SUSPICIOUS_WORDS = ["secure", "verify", "account", "update", "confirm", "portal", "login"]

# Fictional e-signature brand for the device_code lure, so no real organization is named
# (pattern from Truesec's May 2026 device-code report).
ESIGN_BRAND_NAME, ESIGN_BRAND_SLUG = "SignRelay", "signrelay"   # display name, domain slug

# Fictional aged small-business sites for aged device_code lures, which defeat "new domain" checks.
_COMPROMISED_SITE_WORDS = ["riverside", "harbor", "brightwood", "summit", "clearview", "northridge"]
_COMPROMISED_SITE_SUFFIXES = ["bookkeeping", "consulting", "realty", "print", "supply", "studio"]

# Share of emails whose sender has written before. Trusted-source phishing comes from a real
# relationship, so it is close to legitimate mail.
PRIOR_CONTACT_RATE = {"legitimate": 0.85, "trusted_source_phishing": 0.90, "overt_phishing": 0.08}

# Letters and the look-alike digits that replace them in a typosquatted name.
_CONFUSABLES = {"o": "0", "i": "1", "l": "1", "s": "5", "e": "3"}


def typosquat_tenant(tenant: str, rng: random.Random) -> str:
    """Return a lowercase typo variant of a tenant slug (e.g. a doubled or dropped letter).

    It can equal the input when the chosen edit has no effect.
    """
    tenant = tenant.lower()
    if len(tenant) < 3:
        return tenant + tenant[-1]
    technique = rng.choice(["double", "swap", "confusable", "drop", "hyphen"])
    i = rng.randint(1, len(tenant) - 2)
    if technique == "double":
        return tenant[:i] + tenant[i] + tenant[i:]
    if technique == "swap" and i + 1 < len(tenant):
        chars = list(tenant)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if technique == "confusable":
        chars = list(tenant)
        for idx, ch in enumerate(chars):
            if ch in _CONFUSABLES and rng.random() < 0.7:
                chars[idx] = _CONFUSABLES[ch]
                return "".join(chars)
        return tenant[:i] + tenant[i:]  # no confusable letter replaced
    if technique == "drop":
        return tenant[:i] + tenant[i + 1:]
    if technique == "hyphen":
        return tenant[:i] + "-" + tenant[i:]
    return tenant


def make_trusted_source_url(real_tenant: str, rng: random.Random) -> tuple[str, str]:
    """Return (URL, root domain) for a SharePoint/OneDrive link with a typosquatted tenant."""
    root = rng.choice(TRUSTED_URL_ROOTS)
    fake_tenant = typosquat_tenant(real_tenant, rng)
    doc_id = "".join(rng.choices(string.ascii_letters + string.digits, k=22))
    if root == "sharepoint.com":
        url = f"https://{fake_tenant}-my.sharepoint.com/:w:/g/personal/{doc_id}/document"
    else:
        url = f"https://onedrive.live.com/redir?resid={doc_id}&authkey=%21{fake_tenant.upper()}"
    return url, root


def make_device_code_lure_url(rng: random.Random, aged: bool) -> tuple[str, str]:
    """Return (URL, scored domain) for a device_code lure page.

    Only this URL is stored; a SOC pipeline would not see the Microsoft page it hands off to.

    Args:
        aged: True uses an aged small-business domain (enrich with force_dirty=False);
            False uses a new brand look-alike domain (enrich with force_dirty=True).
    """
    if aged:
        word = rng.choice(_COMPROMISED_SITE_WORDS)
        suffix = rng.choice(_COMPROMISED_SITE_SUFFIXES)
        domain = f"{word}{suffix}.com"
    else:
        fake_brand = typosquat_tenant(ESIGN_BRAND_SLUG, rng)
        tld = rng.choice(SUSPICIOUS_TLDS)
        domain = f"{fake_brand}-verify.{tld}"
    doc_id = "".join(rng.choices(string.ascii_letters + string.digits, k=16))
    url = f"https://{domain}/review/{doc_id}"
    return url, domain


def simulate_prior_contact(scenario: str, rng: random.Random) -> bool:
    """Draw whether the sender has written before (rate from PRIOR_CONTACT_RATE, else 0.5)."""
    return rng.random() < PRIOR_CONTACT_RATE.get(scenario, 0.5)


def make_overt_url(rng: random.Random) -> tuple[str, str]:
    """Return (URL, domain) for an overt-phishing link on a suspicious new domain."""
    word = rng.choice(SUSPICIOUS_WORDS)
    junk = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 8)))
    tld = rng.choice(SUSPICIOUS_TLDS)
    domain = f"{word}-{junk}.{tld}"
    path = "".join(rng.choices(string.ascii_lowercase + string.digits, k=10))
    return f"https://{domain}/{path}", domain


def make_legit_url(rng: random.Random, real_tenant: str | None = None) -> tuple[str, str]:
    """Return (URL, root domain) for a genuine SharePoint/OneDrive link ("contoso" if no tenant).

    Legitimate mail needs these links too, or "has a SharePoint link" alone would separate the
    classes. The caller decides whether an email gets a link, so body text and link agree.
    """
    root = rng.choice(TRUSTED_URL_ROOTS)
    tenant = (real_tenant or "contoso").lower()
    doc_id = "".join(rng.choices(string.ascii_letters + string.digits, k=22))
    if root == "sharepoint.com":
        url = f"https://{tenant}-my.sharepoint.com/:w:/g/personal/{doc_id}/document"
    else:
        url = f"https://onedrive.live.com/redir?resid={doc_id}&authkey=%21{tenant.upper()}"
    return url, root


def enrich(scenario: str, url_root_domain: str | None, rng: random.Random,
           force_dirty: bool | None = None) -> dict:
    """Simulate the enrichment fields for one email (deterministic for a given rng).

    Args:
        url_root_domain: The domain the tools would score, or None if there is no link.
        force_dirty: None makes only overt_phishing suspicious; True/False overrides it
            (used by device_code).
    """
    has_url = url_root_domain is not None
    if not has_url:
        return {"has_url": False, "url_domain": "", "domain_age_days": None,
                "dga_score": 0.0, "vt_malicious": 0, "vt_total": 0}

    dirty = (scenario == "overt_phishing") if force_dirty is None else force_dirty

    if dirty:
        age = rng.randint(*NEW_DOMAIN_AGE_RANGE)
        dga = round(rng.uniform(*DIRTY_DGA_RANGE), 3)
        total = rng.randint(*VT_TOTAL_RANGE)
        malicious = min(total, rng.randint(*DIRTY_VT_HITS_RANGE))
    else:
        # Legitimate, typosquat_link and aged device_code share one clean distribution by design.
        age = rng.randint(*OLD_DOMAIN_AGE_RANGE)
        dga = round(rng.uniform(*CLEAN_DGA_RANGE), 3)
        total = rng.randint(*VT_TOTAL_RANGE)
        malicious = 0

    return {"has_url": True, "url_domain": url_root_domain, "domain_age_days": age,
            "dga_score": dga, "vt_malicious": malicious, "vt_total": total}
