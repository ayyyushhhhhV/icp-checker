"""Deterministic helpers: column mapping, company/domain normalization, fuzzy matching, snippet parsing."""

import re
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import unquote, urlsplit

# ---------------------------------------------------------------- column mapping

# Order matters: linkedin/email are matched before website so "LinkedIn URL" isn't taken by "url".
FIELD_ALIASES: dict[str, list[str]] = {
    "name": ["name", "full name", "fullname", "contact name", "contact", "person", "lead name", "first name"],
    "title": ["title", "job title", "designation", "position", "role", "jobtitle"],
    "company": ["company", "company name", "organisation", "organization", "account", "account name", "employer"],
    "linkedin_url": ["linkedin", "linkedin url", "linkedin id", "linkedin handle", "linkedin link", "profile",
                     "linkedin profile", "profile url", "profile link", "li url", "li"],
    "email": ["email", "e-mail", "email address", "work email", "mail"],
    "website": ["website", "domain", "company website", "url", "web", "site", "company domain"],
}


def _norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", h.lower()).strip()


def propose_mapping(columns: list[str]) -> dict[str, str | None]:
    """Map our fields to CSV columns by header matching. Exact alias first, then containment."""
    mapping: dict[str, str | None] = {f: None for f in FIELD_ALIASES}
    used: set[str] = set()
    normed = {c: _norm_header(c) for c in columns}
    for exact in (True, False):
        for field, aliases in FIELD_ALIASES.items():
            if mapping[field]:
                continue
            for col, n in normed.items():
                if col in used:
                    continue
                hit = n in aliases if exact else any(re.search(rf"\b{re.escape(a)}\b", n) for a in aliases)
                if hit:
                    mapping[field] = col
                    used.add(col)
                    break
    return mapping


# ---------------------------------------------------------------- company / domain

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt. ltd", "pvt", "limited", "ltd", "llp", "llc", "inc",
    "incorporated", "corp", "corporation", "co", "company", "gmbh", "plc", "s.a.", "sa", "ag", "bv", "pte",
    "fze", "fzco", "fz-llc", "l.l.c.", "group", "holdings",
]
_SUFFIX_RE = re.compile(r"(?:[\s,.-]+(?:" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r"))+\.?\s*$")


_LEGAL_ONLY = [s for s in LEGAL_SUFFIXES if s not in ("group", "holdings", "company", "co")]
_LEGAL_ONLY_RE = re.compile(
    r"(?:[\s,.-]+(?:" + "|".join(re.escape(s) for s in sorted(_LEGAL_ONLY, key=len, reverse=True)) + r"))+\.?\s*$", re.I
)


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def normalize_company_name(name: str | None) -> str:
    if not name:
        return ""
    s = _ascii(name).lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("&", " and ")
    prev = None
    while prev != s:
        prev = s
        s = _SUFFIX_RE.sub("", s).strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def search_company_name(name: str) -> str:
    """Company name for search queries: legal suffixes removed, original casing kept ("Acme Retail Pvt Ltd" -> "Acme Retail").
    Brand words like "Group"/"Holdings" are kept: "Landmark Group" must not become "Landmark"."""
    s, prev = name.strip(), None
    while prev != s:
        prev = s
        s = _LEGAL_ONLY_RE.sub("", s).strip()
    return s or name.strip()


def normalize_domain(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    if "@" in v and "://" not in v:
        v = v.split("@", 1)[1]
    if "://" not in v:
        v = "https://" + v
    host = urlsplit(v).hostname or ""
    host = host.removeprefix("www.").rstrip(".")
    return host if "." in host else None


FREE_EMAIL = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "rediffmail.com", "proton.me"}

DIRECTORY_DOMAINS = (
    "linkedin.com", "crunchbase.com", "wikipedia.org", "glassdoor", "indeed", "naukri.com", "zoominfo.com",
    "facebook.com", "instagram.com", "x.com", "twitter.com", "youtube.com", "ambitionbox.com", "bloomberg.com",
    "dnb.com", "tracxn.com", "owler.com", "rocketreach.co", "apollo.io", "justdial.com", "indiamart.com",
    "tofler.in", "zaubacorp.com", "economictimes", "craft.co", "pitchbook.com", "g2.com", "builtin.com",
)


def is_directory(domain: str) -> bool:
    return any(d in domain for d in DIRECTORY_DOMAINS)


def company_key(company: str | None, domain: str | None) -> str:
    return f"domain:{domain}" if domain else f"name:{normalize_company_name(company)}"


# ---------------------------------------------------------------- fuzzy matching


def _tokens(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", _ascii(s).lower()).split()


def name_matches(expected: str, candidate: str) -> bool:
    """First and last name tokens of `expected` must appear (fuzzily) in `candidate`."""
    exp, cand = _tokens(expected), _tokens(candidate)
    if not exp or not cand:
        return False
    must = {exp[0], exp[-1]}
    return all(any(t == c or SequenceMatcher(None, t, c).ratio() >= 0.85 for c in cand) for t in must)


def company_matches(expected: str, text: str) -> bool:
    exp = normalize_company_name(expected)
    if not exp:
        return False
    hay = " " + normalize_company_name(text) + " "
    if f" {exp} " in hay or exp.replace(" ", "") in hay.replace(" ", ""):
        return True
    words = hay.split()
    n = len(exp.split())
    return any(SequenceMatcher(None, exp, " ".join(words[i:i + n])).ratio() >= 0.85 for i in range(len(words)))


# ---------------------------------------------------------------- LinkedIn search-result parsing

_LI_SUFFIX = re.compile(r"\s*[|\-–—]\s*linkedin\s*$", re.I)
_SEP = re.compile(r"\s+[-–—|]\s+")


def parse_linkedin_result_title(title: str, company: str | None = None) -> dict | None:
    """Parse Google result titles like 'Jane Doe - Head of TA - Acme Corp | LinkedIn'.

    Returns {"name", "title", "company"} (title/company may be None). Only uses the given text.
    If `company` is given and the 2nd segment is just the company name ('Name - Acme | LinkedIn'),
    it is treated as the company, not a title."""
    if not title:
        return None
    t = _LI_SUFFIX.sub("", title.strip()).strip()
    t = t.replace("...", "").replace("…", "").strip(" -|")
    parts = [p.strip() for p in _SEP.split(t) if p.strip()]
    if not parts:
        return None
    name = _clean_name(parts[0])
    if len(_tokens(name)) < 2 or len(name) > 60:
        return None
    job = _clean_title(parts[1]) if len(parts) >= 2 else None
    if job and " at " in job and len(parts) == 2:
        job, company_part = (x.strip() for x in job.rsplit(" at ", 1))
        return {"name": name, "title": _clean_title(job), "company": company_part}
    if len(parts) == 2 and company and job and company_matches(company, job):
        return {"name": name, "title": None, "company": job}
    return {"name": name, "title": job, "company": parts[2] if len(parts) >= 3 else None}


def _strip_invisible(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch) != "Cf")  # e.g. RTL marks in LinkedIn titles


def _clean_name(s: str) -> str:
    # "Jane Doe, SHRM-SCP, PHR" -> "Jane Doe"; "Dr. Jane Doe" kept as is
    return _strip_invisible(s).split(",")[0].strip(" |-–—·")


def _clean_title(s: str | None) -> str | None:
    if not s:
        return None
    s = _strip_invisible(s).strip(" |-–—·,")
    s = re.sub(r"\s+(at|@)$", "", s, flags=re.I).strip()
    return s or None


_SLUG_RE = re.compile(r"^\w[\w\-.%]{1,99}$")  # \w is Unicode-aware (accented IDs exist)


def linkedin_slug(value: str | None) -> str | None:
    """Accept a profile URL ('https://in.linkedin.com/in/jane-doe-123/'), '/in/jane-doe-123' or a bare ID
    ('jane-doe-123') and return the profile ID. Returns None for company pages or junk."""
    if not value:
        return None
    v = value.strip()
    if m := re.search(r"linkedin\.com/in/([^/?#\s]+)", v, re.I):
        slug = m.group(1)
    elif m := re.fullmatch(r"/?in/([^/?#\s]+)/?", v, re.I):
        slug = m.group(1)
    elif "linkedin.com" in v.lower() or "/" in v.strip("/") or " " in v:
        return None
    else:
        slug = v.strip("/").lstrip("@")
    slug = unquote(slug).strip().lower()
    return slug if _SLUG_RE.match(slug) else None


_EXPERIENCE_RE = re.compile(r"Experience:\s*([^·|\n]+?)\s*(?:·|\||$)")


def company_from_snippet(snippet: str | None) -> str | None:
    """Google's LinkedIn snippets often read 'Experience: Acme Retail · Education: ...'."""
    m = _EXPERIENCE_RE.search(snippet or "")
    return m.group(1).strip() if m else None


def is_linkedin_profile_link(link: str) -> bool:
    return bool(re.match(r"^https?://([a-z]{2,3}\.)?(www\.)?linkedin\.com/in/", link or "", re.I))
