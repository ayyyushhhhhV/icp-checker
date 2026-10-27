"""Person check and stakeholder finding. LinkedIn data comes ONLY from Serper search results
(title/snippet/link); profile pages are never fetched."""

import logging

from .config import get_settings
from .llm import LLM
from .llm_schemas import ProfileSnippetOut, SnippetParseOut, TeamPageOut, TitleNorm, TitleNormalizeOut
from .models import (
    ICP,
    SENIORITY_ORDER,
    CompanyFacts,
    Evidence,
    LeadInput,
    LinkedInProfile,
    PersonFacts,
    Stakeholder,
    StakeholderCandidate,
)
from .normalize import (
    company_from_snippet,
    company_matches,
    is_linkedin_profile_link,
    linkedin_slug,
    name_matches,
    parse_linkedin_result_title,
)
from .scoring import score
from .serper import Budget, SerperBudgetExceeded, SerperClient, SerperError, is_linkedin_url

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- title normalization


async def normalize_titles(titles: list[str], icp: ICP, llm: LLM) -> list[TitleNorm | None]:
    if not titles:
        return []
    out = await llm.structured(
        "title_normalize", TitleNormalizeOut, small=True,
        titles=titles, roles=icp.role_criteria.target_roles, functions=get_settings().functions,
    )
    result: list[TitleNorm | None] = [None] * len(titles)
    for item in out.items if out else []:
        if 0 <= item.index < len(titles):
            result[item.index] = item
    return result


def apply_norm(raw: PersonFacts, norm: TitleNorm | None, icp: ICP) -> PersonFacts:
    p = raw.model_copy(deep=True)
    if norm is None:
        return p
    known = set(get_settings().functions) | {r.function for r in icp.role_criteria.target_roles}
    p.function = norm.function if norm.function in known else ("other" if norm.function else None)
    p.seniority = norm.seniority
    p.role_type = norm.role_type
    p.matched_role_index = norm.matched_role_index
    return p


async def normalize_person(raw: PersonFacts | None, icp: ICP, llm: LLM) -> PersonFacts | None:
    if raw is None or not raw.title:
        return raw
    [norm] = await normalize_titles([raw.title], icp, llm)
    return apply_norm(raw, norm, icp)


# ---------------------------------------------------------------- LinkedIn profile from the list


async def lookup_linkedin_profile(value: str, serper: SerperClient, llm: LLM) -> LinkedInProfile | None:
    """Resolve a LinkedIn URL / ID from the list with ONE Google search (`site:linkedin.com/in/<id>`).
    Only the result title/snippet is used — the profile page itself is never requested."""
    slug = linkedin_slug(value)
    if not slug:
        return None
    profile = LinkedInProfile(slug=slug, url=f"https://www.linkedin.com/in/{slug}")
    try:
        res = await serper.search(f"site:linkedin.com/in/{slug}", num=5)
    except SerperError as e:
        log.info("linkedin lookup failed for %s: %s", slug, e)
        return profile
    hit = next((r for r in res.get("organic", []) if linkedin_slug(r.get("link", "")) == slug), None)
    if not hit:
        return profile
    profile.found, profile.url = True, hit["link"]
    profile.result_title, profile.snippet = hit.get("title", ""), hit.get("snippet", "")
    parsed = parse_linkedin_result_title(profile.result_title) or {}
    profile.name, profile.title = parsed.get("name"), parsed.get("title")
    profile.company = parsed.get("company") or company_from_snippet(profile.snippet)
    if not (profile.name and profile.title and profile.company):  # odd format -> let the LLM read the same text
        out = await llm.structured("profile_snippet_parse", ProfileSnippetOut, small=True,
                                   result_title=profile.result_title, snippet=profile.snippet)
        if out:
            profile.name = profile.name or out.name
            profile.title = profile.title or out.title
            profile.company = profile.company or out.company
    return profile


# ---------------------------------------------------------------- person check (ICP-independent)


def _li_search_query(name: str, company: str) -> str:
    return f'site:linkedin.com/in "{name}" "{company}"'


async def check_person(
    inp: LeadInput, company: str, serper: SerperClient | None, budget: Budget, profile: LinkedInProfile | None = None,
) -> PersonFacts | None:
    """Take name/title from the list; verify or fill the title from one LinkedIn-site search (snippets only).
    If the list gave a LinkedIn profile that was found, that result is used instead of a name search.
    Pass serper=None to skip the search."""
    name = inp.name or (profile.name if profile and profile.found else None)
    if not name and not inp.title:
        return None
    p = PersonFacts(name=name, title=inp.title, title_source="list" if inp.title else None)
    if inp.title:
        p.evidence.append(Evidence(fact=f"Title from uploaded list: {inp.title}", source_type="list"))
    if profile and profile.found and (not inp.name or name_matches(inp.name, profile.name or "")):
        p.verified, p.linkedin_url = True, profile.url
        p.evidence.append(Evidence(fact=f"LinkedIn search result: {profile.result_title}", source_url=profile.url,
                                   source_type="search_snippet"))
        if not p.title and profile.title:
            p.title, p.title_source = profile.title, "search_snippet"
        return p
    if serper is None or not (name and company):
        return p
    try:
        res = await serper.search(_li_search_query(name, company), num=10, budget=budget)
    except (SerperBudgetExceeded, SerperError) as e:
        log.info("person verification skipped: %s", e)
        return p
    for r in res.get("organic", []):
        link, title, snippet = r.get("link", ""), r.get("title", ""), r.get("snippet", "")
        if not is_linkedin_profile_link(link):
            continue
        parsed = parse_linkedin_result_title(title, company)
        if not parsed or not name_matches(name, parsed["name"]):
            continue
        if not company_matches(company, f"{parsed.get('company') or ''} {title} {snippet}"):
            continue
        p.verified = True
        p.linkedin_url = link
        p.evidence.append(Evidence(fact=f"Search result: {title}", source_url=link, source_type="search_snippet"))
        if not p.title and parsed.get("title"):
            p.title, p.title_source = parsed["title"], "search_snippet"
        break
    return p if (p.title or p.name) else None


# ---------------------------------------------------------------- stakeholder finding


def _role_titles(icp: ICP, max_roles: int = 3, per_role: int = 2) -> list[str]:
    titles: list[str] = []
    for role in icp.role_criteria.target_roles[:max_roles]:
        titles.extend(role.title_examples[:per_role])
    return titles


def search_titles(icp: ICP) -> list[str]:
    return _role_titles(icp, max_roles=3, per_role=3)


async def find_candidates(
    company: str, domain: str | None, icp: ICP, serper: SerperClient, llm: LLM, budget: Budget,
) -> list[StakeholderCandidate]:
    found: dict[str, StakeholderCandidate] = {}
    titles = _role_titles(icp)
    try:
        if titles:
            q = "site:linkedin.com/in (" + " OR ".join(f'"{t}"' for t in titles) + f') "{company}"'
            res = await serper.search(q, num=10, budget=budget)
            results = [r for r in res.get("organic", []) if is_linkedin_profile_link(r.get("link", ""))]
            for r in results:
                parsed = parse_linkedin_result_title(r.get("title", ""), company)
                if not parsed or not parsed.get("title"):
                    continue
                if not company_matches(company, f"{parsed.get('company') or ''} {r.get('title', '')} {r.get('snippet', '')}"):
                    continue
                found.setdefault(parsed["name"].lower(), StakeholderCandidate(
                    name=parsed["name"], title=parsed["title"], linkedin_url=r["link"],
                    source="search_snippet", source_url=r["link"],
                ))
            if not found and results:  # titles in odd formats -> let the LLM read the result text
                out = await llm.structured("snippet_parse", SnippetParseOut, small=True, company_name=company, results=results)
                links = {r["link"] for r in results}
                for person in out.people if out else []:
                    if person.link in links and person.title and company_matches(company, person.company or ""):
                        found.setdefault(person.name.lower(), StakeholderCandidate(
                            name=person.name, title=person.title, linkedin_url=person.link,
                            source="search_snippet", source_url=person.link,
                        ))

        if not found and domain:  # fallback: company's own leadership / team page
            res = await serper.search(f'site:{domain} leadership OR team OR "about us"', num=5, budget=budget)
            url = next((r["link"] for r in res.get("organic", []) if r.get("link") and not is_linkedin_url(r["link"])), None)
            if url:
                page = await serper.scrape(url, budget=budget)
                text = (page.get("markdown") or page.get("text") or "")[: get_settings().page_char_budget]
                out = await llm.structured("team_page_extract", TeamPageOut, small=True, company_name=company, url=url, text=text)
                for m in out.people if out else []:
                    found.setdefault(m.name.lower(), StakeholderCandidate(
                        name=m.name, title=m.title, source="company_website", source_url=url,
                    ))
    except (SerperBudgetExceeded, SerperError) as e:
        log.info("stakeholder search stopped early for %s: %s", company, e)
    return list(found.values())


def _relation(lead: PersonFacts | None, s: PersonFacts) -> str | None:
    if not lead or not lead.function or lead.function != s.function:
        return None
    if lead.seniority and s.seniority and SENIORITY_ORDER.index(s.seniority) > SENIORITY_ORDER.index(lead.seniority):
        return "Same department, more senior (likely manager or above)"
    return "Same department"


async def build_stakeholders(
    candidates: list[StakeholderCandidate], facts: CompanyFacts, icp: ICP, llm: LLM, lead: PersonFacts | None,
) -> list[Stakeholder]:
    if lead and lead.name:
        candidates = [c for c in candidates if not name_matches(lead.name, c.name)]
    if not candidates:
        return []
    norms = await normalize_titles([c.title or "" for c in candidates], icp, llm)
    out: list[Stakeholder] = []
    for cand, norm in zip(candidates, norms):
        src_type = "search_snippet" if cand.source == "search_snippet" else "website"
        ev = Evidence(fact=f"{cand.name} — {cand.title}", source_url=cand.source_url, source_type=src_type)
        person = apply_norm(PersonFacts(
            name=cand.name, title=cand.title, title_source="search_snippet", linkedin_url=cand.linkedin_url, evidence=[ev],
        ), norm, icp)
        sc = score(facts, person, icp, has_stakeholders=False)
        if sc.person_fit.points <= 0:
            continue
        out.append(Stakeholder(
            name=cand.name, title=cand.title, linkedin_url=cand.linkedin_url, source=cand.source,
            function=person.function, seniority=person.seniority, role_type=person.role_type,
            score=sc, relation_to_lead=_relation(lead, person), evidence=[ev],
        ))
    out.sort(key=lambda s: (-s.score.total, s.name))
    return out[: get_settings().max_stakeholders]
