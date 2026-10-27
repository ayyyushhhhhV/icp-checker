"""Company research (Serper only) and fact extraction (LLM), kept separate so facts can be
re-extracted for a new ICP version from stored sources without new Serper calls."""

import logging
import re
from urllib.parse import urljoin, urlsplit

from .config import get_settings
from .llm import LLM
from .llm_schemas import CompanyExtractOut, DomainPickOut
from .models import ICP, CompanyFacts, CompanyResearch, CompanySignal, Evidence, SourceDoc
from .normalize import company_matches, is_directory, normalize_company_name, normalize_domain
from .serper import Budget, LinkedInFetchForbidden, SerperBudgetExceeded, SerperClient, SerperError, is_linkedin_url
from .store import get_company_research, save_company_research

log = logging.getLogger(__name__)

_MD_LINK = re.compile(r"\[([^\]]{0,80})\]\((https?://[^)\s]+|/[^)\s]*)\)")
CAREERS_HINTS = ("career", "jobs", "join-us", "join us", "work-with-us", "work with us", "hiring", "openings")
ABOUT_HINTS = ("about", "who-we-are", "who we are", "our-story", "company")


def _same_site(url: str, domain: str) -> bool:
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    return host == domain or host.endswith("." + domain)


_ASSET_EXT = re.compile(r"\.(jpe?g|png|gif|webp|svg|ico|pdf|mp4|webm|zip|css|js)(\?|$)", re.I)


def find_links(markdown: str, base_url: str, domain: str) -> dict[str, str | None]:
    """Find careers/about links in homepage markdown. Careers may be on an ATS host (lever, greenhouse...)."""
    found: dict[str, str | None] = {"careers": None, "about": None}
    for text, href in _MD_LINK.findall(markdown or ""):
        url = urljoin(base_url, href).split("#")[0]
        if is_linkedin_url(url) or _ASSET_EXT.search(urlsplit(url).path + ("?" if urlsplit(url).query else "")):
            continue
        hay = f"{text} {urlsplit(url).path}".lower()
        if not found["careers"] and any(h in hay for h in CAREERS_HINTS):
            found["careers"] = url
        elif not found["about"] and _same_site(url, domain) and any(h in hay for h in ABOUT_HINTS):
            found["about"] = url
    return found


def _trim(text: str, budget: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # drop images
    return text[:budget]


async def resolve_domain(name: str, serper: SerperClient, llm: LLM, budget: Budget) -> tuple[str | None, str | None]:
    res = await serper.search(f'"{name}" official website', num=10, budget=budget)
    kg_site = normalize_domain((res.get("knowledgeGraph") or {}).get("website"))
    if kg_site and not is_directory(kg_site):
        return kg_site, (res.get("knowledgeGraph") or {}).get("website")
    organic = [r for r in res.get("organic", []) if r.get("link")]
    norm = normalize_company_name(name).replace(" ", "")
    for r in organic[:5]:
        d = normalize_domain(r["link"])
        if d and not is_directory(d) and norm and norm[:8] in d.replace("-", "").replace(".", ""):
            return d, r["link"]
    candidates = [r for r in organic if not is_directory(normalize_domain(r["link"]) or "")][:8]
    if not candidates:
        return None, None
    out = await llm.structured("domain_pick", DomainPickOut, small=True, company_name=name, results=candidates)
    if out and out.domain:
        d = normalize_domain(out.domain)
        if d and not is_directory(d):
            return d, out.source_url
    return None, None


async def research_company(
    name: str, domain: str | None, key: str, serper: SerperClient, llm: LLM, budget: Budget
) -> CompanyResearch:
    """Gather source text for a company. Cached by company key (default TTL 30 days)."""
    s = get_settings()
    cached = get_company_research(key, ttl_days=s.company_cache_ttl_days)
    if cached:
        return cached

    r = CompanyResearch(key=key, name=name, domain=domain)
    try:
        if not r.domain:
            r.domain, src = await resolve_domain(name, serper, llm, budget)
            r.notes.append(f"domain resolved via search: {r.domain} ({src})" if r.domain else "domain not found")

        # 1) website pages (max 3)
        if r.domain:
            home_url = f"https://{r.domain}"
            pages: list[tuple[str, str]] = [(home_url, "website")]
            home_md = await _scrape_into(r, home_url, "website", serper, budget)
            links = find_links(home_md or "", home_url, r.domain)
            if not links["careers"]:
                try:
                    res = await serper.search(f"site:{r.domain} careers", num=5, budget=budget)
                    organic = [x for x in res.get("organic", []) if x.get("link") and not is_linkedin_url(x["link"])]
                    links["careers"] = organic[0]["link"] if organic else None
                    for x in organic[:5]:  # snippets often state hiring volume ("500+ openings") even when the page is JS-only
                        r.sources.append(SourceDoc(url=x["link"], source_type="careers", title=x.get("title", ""),
                                                   text=x.get("snippet", "")))
                except SerperBudgetExceeded:
                    pass
            for kind, url in (("careers", links["careers"]), ("website", links["about"])):
                if url and len(pages) < s.max_pages_per_company and url not in {p[0] for p in pages}:
                    pages.append((url, kind))
                    await _scrape_into(r, url, kind, serper, budget)

        # 2) news, last 6 months (titles + snippets only)
        news = await serper.news(f'"{name}"', num=10, tbs=s.news_time_filter, budget=budget)
        items = [n for n in news.get("news", []) if company_matches(name, f"{n.get('title', '')} {n.get('snippet', '')}")]
        for n in items[: s.news_results]:
            r.sources.append(SourceDoc(
                url=n.get("link", ""), source_type="news", title=n.get("title", ""),
                text=f"{n.get('date', '')} — {n.get('source', '')}: {n.get('snippet', '')}",
            ))

        # 3) size hint (snippets only — LinkedIn links are never opened)
        size = await serper.search(f'"{name}" employees', num=10, budget=budget)
        kg = size.get("knowledgeGraph") or {}
        if kg:
            attrs = "; ".join(f"{k}: {v}" for k, v in (kg.get("attributes") or {}).items())
            r.sources.append(SourceDoc(
                url=kg.get("website") or kg.get("descriptionLink") or "google-knowledge-graph",
                source_type="search_snippet", title=kg.get("title", ""),
                text=f"{kg.get('type', '')}. {kg.get('description', '')} {attrs}".strip(),
            ))
        for o in size.get("organic", [])[:6]:
            r.sources.append(SourceDoc(
                url=o.get("link", ""), source_type="search_snippet", title=o.get("title", ""), text=o.get("snippet", ""),
            ))
    except SerperBudgetExceeded as e:
        r.notes.append(str(e))
    except SerperError as e:
        r.notes.append(f"serper error: {e}")
        r.serper_calls = budget.used
        return r  # not cached, so a rerun can retry

    r.serper_calls = budget.used
    save_company_research(r)
    return r


async def _scrape_into(r: CompanyResearch, url: str, kind: str, serper: SerperClient, budget: Budget) -> str | None:
    try:
        page = await serper.scrape(url, budget=budget)
    except LinkedInFetchForbidden:
        return None
    except SerperError as e:
        r.notes.append(f"scrape failed {url}: {e}")
        return None
    md = page.get("markdown") or page.get("text") or ""
    if len(md.strip()) < 200:
        r.notes.append(f"scrape returned little text: {url}")
    r.sources.append(SourceDoc(
        url=url, source_type=kind, title=(page.get("metadata") or {}).get("title", ""),
        text=_trim(md, get_settings().page_char_budget),
    ))
    r.pages_scraped += 1
    return md


async def extract_company_facts(research: CompanyResearch, icp: ICP, llm: LLM) -> CompanyFacts:
    """LLM extraction (structured) -> CompanyFacts. Drops any fact whose source_url wasn't provided."""
    facts = CompanyFacts(key=research.key, name=research.name, domain=research.domain)
    website_chars = sum(len(d.text) for d in research.sources if d.source_type in ("website", "careers"))
    facts.thin_data = website_chars < 500
    if not research.sources:
        return facts

    out = await llm.structured(
        "company_extract", CompanyExtractOut,
        company_name=research.name, domain=research.domain, icp=icp, sources=research.sources,
    )
    if out is None:
        facts.thin_data = True
        return facts

    known = {d.url: d.source_type for d in research.sources}

    def ev(fact: str, url: str | None) -> Evidence | None:
        if not url or url not in known:
            return None
        return Evidence(fact=fact, source_url=url, source_type=known[url])

    if out.industry and (e := ev(f"Industry: {out.industry.value}", out.industry.source_url)):
        facts.industry = out.industry.value
        facts.evidence.append(e)
        facts.industry_match = out.industry_match
    elif out.industry_match == "excluded":
        facts.industry_match = "unknown"  # never disqualify on an unsourced claim
    if out.sub_industry and ev("", out.sub_industry.source_url):
        facts.sub_industry = out.sub_industry.value
    if out.employee_estimate is not None and (
        e := ev(f"Employees: ~{out.employee_estimate} ({out.employee_range or 'estimate'})", out.employee_source_url)
    ):
        facts.employee_estimate = out.employee_estimate
        facts.employee_range = out.employee_range
        facts.size_confidence = out.size_confidence
        facts.evidence.append(e)
    for loc in out.locations:
        if e := ev(f"Location: {loc.value}", loc.source_url):
            facts.locations.append(loc.value)
            facts.evidence.append(e)
    if facts.locations:
        facts.matched_locations = [x for x in out.matched_icp_locations if x in icp.company_criteria.locations]
        facts.excluded_location_hits = [
            x for x in out.matched_excluded_locations if x in icp.company_criteria.excluded_locations
        ]
    wanted = icp.company_criteria.need_signals
    for sig in out.signals:
        if not 0 <= sig.signal_index < len(wanted):
            continue
        name = wanted[sig.signal_index].signal
        if e := ev(f"{name}: {sig.detail}", sig.source_url):
            facts.signals.append(CompanySignal(signal=name, detail=sig.detail, evidence=e))
            facts.evidence.append(e)
    return facts
