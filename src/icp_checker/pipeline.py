"""Background job: group leads per company, research once, gate, check people, find stakeholders,
score, explain. Rows are saved as they finish."""

import asyncio
import csv
import io
import logging
from collections import Counter, defaultdict

from .config import get_settings
from .llm import LLM
from .llm_schemas import ExplainOut
from .models import ICP, CompanyFacts, CompanyResearch, Lead, LeadInput, LeadList, Score, Usage
from .normalize import (
    FREE_EMAIL,
    company_key,
    linkedin_slug,
    normalize_company_name,
    normalize_domain,
    search_company_name,
)
from .people import (
    build_stakeholders,
    check_person,
    find_candidates,
    lookup_linkedin_profile,
    normalize_person,
    search_titles,
)
from .research import extract_company_facts, research_company
from .scoring import needs_stakeholders, score
from .serper import Budget, SerperClient
from .store import (
    get_company_research,
    get_leads,
    get_list,
    save_leads,
    save_list,
    track_usage,
    upsert_lead,
)

log = logging.getLogger(__name__)
_jobs: set[asyncio.Task] = set()
_list_locks: dict[str, asyncio.Lock] = {}


# ---------------------------------------------------------------- list -> leads


def lead_inputs(lst: LeadList) -> list[tuple[int, LeadInput]]:
    """Apply the confirmed column mapping and drop duplicate rows."""
    seen: set[tuple] = set()
    out = []
    for i, row in enumerate(lst.rows):
        vals = {f: (row.get(col) or "").strip() or None for f, col in lst.mapping.items() if col}
        inp = LeadInput(**vals)
        dedupe = (
            (inp.email or "").lower()
            or linkedin_slug(inp.linkedin_url)
            or ((inp.name or "").lower(), normalize_company_name(inp.company), (inp.title or "").lower())
        )
        if not any([inp.name, inp.company, inp.website, inp.email, inp.linkedin_url]) or dedupe in seen:
            continue
        seen.add(dedupe)
        out.append((i, inp))
    return out


def effective_company(lead: Lead) -> str | None:
    """Company from the list, else from the LinkedIn search result for the row's profile."""
    if lead.input.company:
        return lead.input.company
    p = lead.linkedin_profile
    return p.company if p and p.found else None


def lead_domain(inp: LeadInput) -> str | None:
    if d := normalize_domain(inp.website):
        return d
    if inp.email and "@" in inp.email:
        d = normalize_domain(inp.email)
        if d and d not in FREE_EMAIL:
            return d
    return None


# ---------------------------------------------------------------- per-lead scoring


async def explain(lead: Lead, facts: CompanyFacts, icp: ICP, llm: LLM) -> None:
    sc = lead.score
    evidence = (facts.evidence + (lead.person.evidence if lead.person else []))[:10]
    try:
        out = await llm.structured(
            "explain", ExplainOut, small=True,
            company_name=facts.name, lead_name=lead.input.name or (lead.person.name if lead.person else None), lead_title=lead.person.title if lead.person else None,
            score=sc, facts=facts, evidence=evidence, stakeholders=lead.stakeholders, search_titles=search_titles(icp),
        )
    except Exception as e:  # explanation is cosmetic; never fail the row on it
        log.warning("explain failed: %s", e)
        out = None
    if out:
        sc.explanation, sc.recommendation = out.explanation, out.recommendation
    else:
        sc.explanation = f"Category {sc.category} with {sc.total}/100 (company {sc.company_fit.points}/{sc.company_fit.max}, person {sc.person_fit.points}/{sc.person_fit.max})."
        if sc.category == "contact_not_found":
            sc.recommendation = "Search for: " + ", ".join(search_titles(icp))
    if sc.category == "not_a_fit" and sc.disqualified_reason and not sc.recommendation:
        sc.recommendation = f"Skip — {sc.disqualified_reason}"


def _keep_override(old: Score | None, new: Score) -> Score:
    if old and old.overridden:
        new.overridden, new.override = True, old.override
    return new


async def score_lead(lead: Lead, facts: CompanyFacts, icp: ICP, llm: LLM, find_candidates_fn=None) -> Lead:
    """Score one lead from stored raw facts. `find_candidates_fn` is None for offline re-scoring."""
    company_gate = score(facts, None, icp, has_stakeholders=True)
    lead.icp_id, lead.icp_version, lead.company = icp.id, icp.version, facts
    if company_gate.category == "not_a_fit":
        lead.person, lead.stakeholders = None, []  # early exit: no person / stakeholder work
        lead.score = _keep_override(lead.score, score(facts, None, icp, has_stakeholders=False))
    else:
        lead.person = await normalize_person(lead.raw_person, icp, llm)
        lead.stakeholders = []
        if needs_stakeholders(lead.person, icp):
            if lead.stakeholder_candidates is None and find_candidates_fn is not None:
                lead.stakeholder_candidates = await find_candidates_fn()
            lead.stakeholders = await build_stakeholders(lead.stakeholder_candidates or [], facts, icp, llm, lead.person)
        lead.score = _keep_override(lead.score, score(facts, lead.person, icp, has_stakeholders=bool(lead.stakeholders)))
    await explain(lead, facts, icp, llm)
    lead.status = "partial" if facts.thin_data else "scored"
    return lead


# ---------------------------------------------------------------- job


async def _bump_progress(list_id: str, usage: Usage | None = None) -> None:
    async with _list_locks.setdefault(list_id, asyncio.Lock()):
        lst = get_list(list_id)
        lst.progress["done"] += 1
        if usage:
            lst.usage = usage
        save_list(lst)


async def _process_company(
    leads: list[Lead], icp: ICP, serper: SerperClient, llm: LLM, list_usage: Usage, rescore: bool,
) -> None:
    first = leads[0]
    raw_name = next((c for l in leads if (c := effective_company(l))), None)
    name = search_company_name(raw_name) if raw_name else (lead_domain(first.input) or "unknown")
    domain = next((d for l in leads if (d := lead_domain(l.input))), None)
    budget = Budget(get_settings().max_serper_calls_per_company)
    try:
        if rescore:
            research = get_company_research(first.company_key) or CompanyResearch(key=first.company_key, name=name)
        else:
            research = await research_company(name, domain, first.company_key, serper, llm, budget)
        facts = await extract_company_facts(research, icp, llm)
    except Exception as e:
        log.exception("company research failed: %s", name)
        for lead in leads:
            lead.status, lead.error = "failed", f"company research failed: {e}"
            await upsert_lead(lead)
            await _bump_progress(lead.list_id, list_usage)
        return

    # stakeholders are searched once per company, lazily, and shared by its leads
    shared: dict[str, object] = {}

    async def find_once():
        if "c" not in shared:
            shared["c"] = await find_candidates(name, research.domain, icp, serper, llm, budget)
        return shared["c"]

    for lead in leads:
        lead_usage = Usage()
        try:
            with track_usage(lead_usage):
                if not rescore:
                    lead.status = "researching"
                    # not-a-fit companies skip the Serper person search (list data is still kept for re-scoring)
                    fits = score(facts, None, icp, has_stakeholders=True).category != "not_a_fit"
                    lead.raw_person = await check_person(
                        lead.input, name, serper if fits else None, budget, lead.linkedin_profile
                    )
                await score_lead(lead, facts, icp, llm, None if rescore else find_once)
        except Exception as e:
            log.exception("lead failed row=%s", lead.row_index)
            lead.status, lead.error = "failed", str(e)
        lead.usage = lead_usage.model_dump()
        await upsert_lead(lead)
        await _bump_progress(lead.list_id, list_usage)


async def run_list(
    list_id: str, icp: ICP, rescore: bool = False, serper: SerperClient | None = None, llm: LLM | None = None,
) -> None:
    s = get_settings()
    lst = get_list(list_id)
    usage = Usage()
    if rescore:
        leads = get_leads(list_id)
        for l in leads:
            l.status = "queued"
    else:
        leads = [Lead(list_id=list_id, row_index=i, input=inp) for i, inp in lead_inputs(lst)]
    save_leads(list_id, leads)
    lst.status, lst.icp_id, lst.icp_version = "running", icp.id, icp.version
    lst.progress = {"total": len(leads), "done": 0}
    save_list(lst)

    serper = serper or SerperClient(offline=rescore)
    llm = llm or LLM()

    if not rescore:
        # Rows with a LinkedIn URL / ID: one search-snippet lookup each (name, title, company). Never a page fetch.
        sem_li = asyncio.Semaphore(s.max_concurrent_companies)

        async def resolve(lead: Lead):
            async with sem_li:
                lead.linkedin_profile = await lookup_linkedin_profile(lead.input.linkedin_url, serper, llm)

        with track_usage(usage):
            await asyncio.gather(*(resolve(l) for l in leads if linkedin_slug(l.input.linkedin_url)))
        for l in leads:
            company, domain = effective_company(l), lead_domain(l.input)
            l.company_key = company_key(company, domain) if (company or domain) else None
    for l in leads:
        if not l.company_key:
            lp = l.linkedin_profile
            l.status, l.error = "failed", (
                "LinkedIn profile not found via search; add the company to this row"
                if lp and not lp.found
                else "LinkedIn search result doesn't show a current company; add the company to this row"
                if lp
                else "row has no company, website, work email or LinkedIn profile"
            )
    save_leads(list_id, leads)
    lst.progress = {"total": len(leads), "done": sum(1 for l in leads if l.status == "failed")}
    lst.error = None
    save_list(lst)

    groups: dict[str, list[Lead]] = defaultdict(list)
    for l in leads:
        if l.company_key:
            groups[l.company_key].append(l)

    sem = asyncio.Semaphore(s.max_concurrent_companies)

    async def worker(group: list[Lead]):
        async with sem:
            await _process_company(group, icp, serper, llm, usage, rescore)

    try:
        with track_usage(usage):
            await asyncio.gather(*(worker(g) for g in groups.values()))
        status, error = "done", None
    except Exception as e:
        log.exception("list job failed")
        status, error = "failed", str(e)
    finally:
        await serper.aclose()
    lst = get_list(list_id)
    lst.status, lst.error, lst.usage = status, error, usage
    save_list(lst)


def start_job(list_id: str, icp: ICP, rescore: bool = False) -> None:
    task = asyncio.create_task(run_list(list_id, icp, rescore))
    _jobs.add(task)
    task.add_done_callback(_jobs.discard)


# ---------------------------------------------------------------- output


def final_category(l: Lead) -> str | None:
    if not l.score:
        return None
    return l.score.override.category if l.score.overridden and l.score.override else l.score.category


def summary(list_id: str) -> dict:
    leads = get_leads(list_id)
    counts = Counter(final_category(l) or l.status for l in leads)
    scored = [l for l in leads if l.score]
    return {
        "counts": dict(counts),
        "pct_not_a_fit": round(100 * counts.get("not_a_fit", 0) / len(scored), 1) if scored else None,
        "overrides": sum(1 for l in leads if l.score and l.score.overridden),
    }


CATEGORY_ORDER = {"ideal": 0, "good": 1, "wrong_contact": 2, "contact_not_found": 3, "not_a_fit": 4}


def ranked(leads: list[Lead]) -> list[Lead]:
    return sorted(leads, key=lambda l: (CATEGORY_ORDER.get(final_category(l) or "", 9), -(l.score.total if l.score else -1), l.row_index))


def export_csv(lst: LeadList) -> str:
    leads = ranked(get_leads(lst.id))
    buf = io.StringIO()
    extra = ["Category", "Score", "Confidence", "Explanation", "Recommendation", "Evidence links"]
    for i in range(1, 4):
        extra += [f"Stakeholder {i}", f"Stakeholder {i} title", f"Stakeholder {i} link"]
    extra += ["LinkedIn lookup", "Status", "Original category", "Override reason"]
    w = csv.writer(buf)
    w.writerow(lst.columns + extra)
    for l in leads:
        row = lst.rows[l.row_index]
        sc = l.score
        evidence = (l.company.evidence if l.company else []) + (l.person.evidence if l.person else [])
        links = list(dict.fromkeys(e.source_url for e in evidence if e.source_url))[:6]
        out = [
            final_category(l) or "", sc.total if sc else "", sc.confidence if sc else "",
            sc.explanation if sc else (l.error or ""), sc.recommendation if sc else "", " ".join(links),
        ]
        for i in range(3):
            st = l.stakeholders[i] if i < len(l.stakeholders) else None
            out += [st.name, st.title or "", st.linkedin_url or ""] if st else ["", "", ""]
        lp = l.linkedin_profile
        out += [" — ".join(x for x in (lp.name, lp.title, lp.company) if x) if lp and lp.found else ("not found" if lp else "")]
        out += [l.status, sc.category if sc and sc.overridden else "", sc.override.reason if sc and sc.overridden else ""]
        w.writerow([row.get(c, "") for c in lst.columns] + out)
    return buf.getvalue()
