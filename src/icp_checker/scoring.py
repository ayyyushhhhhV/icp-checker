"""Deterministic rules engine. Pure functions: no I/O, no LLM. Same inputs -> same score."""

from .models import (
    ICP,
    SENIORITY_ORDER,
    CompanyFacts,
    Confidence,
    FitPart,
    PersonFacts,
    Score,
    TargetRole,
)

INFLUENCER_FACTOR = 0.6
RELATED_INDUSTRY_FACTOR = 0.5
OTHER_FUNCTION_FACTOR = 0.25
SIZE_TOLERANCE = 0.25


def _r(x: float) -> float:
    return round(x, 2)


# ---------------------------------------------------------------- company criteria


def industry_points(c: CompanyFacts, icp: ICP) -> tuple[float, str | None]:
    w = icp.weights.industry
    if c.industry_match == "excluded":
        return 0, f"Excluded industry ({c.industry or 'flagged'})"
    if not icp.company_criteria.industries:
        return w, None  # ICP doesn't constrain industry
    return {"exact": w, "related": w * RELATED_INDUSTRY_FACTOR}.get(c.industry_match, 0), None


def size_points(c: CompanyFacts, icp: ICP) -> float:
    w = icp.weights.size
    lo, hi = icp.company_criteria.size_min_employees, icp.company_criteria.size_max_employees
    if lo is None and hi is None:
        return w
    n = c.employee_estimate
    if n is None:
        return 0
    if (lo is None or n >= lo) and (hi is None or n <= hi):
        return w
    near_lo = lo is None or n >= lo * (1 - SIZE_TOLERANCE)
    near_hi = hi is None or n <= hi * (1 + SIZE_TOLERANCE)
    return w * 0.5 if near_lo and near_hi else 0


def location_points(c: CompanyFacts, icp: ICP) -> tuple[float, str | None]:
    w = icp.weights.location
    excluded = [x for x in c.excluded_location_hits if x in icp.company_criteria.excluded_locations]
    if excluded:
        return 0, f"Excluded location ({', '.join(excluded)})"
    if not icp.company_criteria.locations:
        return w, None
    matched = [x for x in c.matched_locations if x in icp.company_criteria.locations]
    return (w if matched else 0), None


def need_signal_points(c: CompanyFacts, icp: ICP) -> float:
    w = icp.weights.need_signals
    wanted = {s.signal for s in icp.company_criteria.need_signals}
    if not wanted:
        return w
    present = {s.signal for s in c.signals if s.evidence.source_url} & wanted
    return w * min(1.0, len(present) / len(wanted))


def company_fit(c: CompanyFacts, icp: ICP) -> tuple[FitPart, str | None]:
    ind, dq1 = industry_points(c, icp)
    loc, dq2 = location_points(c, icp)
    breakdown = {
        "industry": _r(ind),
        "size": _r(size_points(c, icp)),
        "location": _r(loc),
        "need_signals": _r(need_signal_points(c, icp)),
    }
    return FitPart(points=_r(sum(breakdown.values())), max=icp.weights.company, breakdown=breakdown), dq1 or dq2


# ---------------------------------------------------------------- person criteria


def matched_role(p: PersonFacts, icp: ICP) -> TargetRole | None:
    roles = icp.role_criteria.target_roles
    if p.matched_role_index is not None and 0 <= p.matched_role_index < len(roles):
        if roles[p.matched_role_index].function == p.function:
            return roles[p.matched_role_index]
    return next((r for r in roles if r.function == p.function), None)


def function_points(p: PersonFacts, icp: ICP) -> float:
    w = icp.weights.function
    if not p.function:
        return 0
    if p.function in {r.function for r in icp.role_criteria.target_roles}:
        return w
    if p.function in icp.role_criteria.deprioritized_functions:
        return 0
    return w * OTHER_FUNCTION_FACTOR


def seniority_points(p: PersonFacts, icp: ICP) -> float:
    w = icp.weights.seniority
    if not p.seniority or p.seniority in icp.role_criteria.excluded_seniority:
        return 0
    role = matched_role(p, icp)
    if role is None:
        return 0
    gap = SENIORITY_ORDER.index(p.seniority) - SENIORITY_ORDER.index(role.seniority_min)
    return w if gap >= 0 else w * 0.5 if gap == -1 else 0


def role_type_points(p: PersonFacts, icp: ICP) -> float:
    return icp.weights.role_type * {"buyer": 1.0, "influencer": INFLUENCER_FACTOR}.get(p.role_type, 0.0)


def person_fit(p: PersonFacts | None, icp: ICP) -> FitPart:
    if p is None or not p.title:
        breakdown = {"function": 0.0, "seniority": 0.0, "role_type": 0.0}
    else:
        breakdown = {
            "function": _r(function_points(p, icp)),
            "seniority": _r(seniority_points(p, icp)),
            "role_type": _r(role_type_points(p, icp)),
        }
    return FitPart(points=_r(sum(breakdown.values())), max=icp.weights.person, breakdown=breakdown)


# ---------------------------------------------------------------- confidence + category


def confidence(c: CompanyFacts, p: PersonFacts | None) -> Confidence:
    if c.thin_data:
        return "low"
    level: Confidence = "high" if p is not None and p.title_source == "rep_capture" else "medium"
    if c.employee_estimate is None and level == "medium":
        level = "low"  # size unknown lowers confidence
    return level


def has_person(p: PersonFacts | None) -> bool:
    return p is not None and bool(p.title)


def score(c: CompanyFacts, p: PersonFacts | None, icp: ICP, has_stakeholders: bool = False) -> Score:
    cfit, dq = company_fit(c, icp)
    pfit = person_fit(p, icp)
    total = int(round(cfit.points + pfit.points))
    t = icp.thresholds

    if dq or cfit.pct < t.company_fit_min_pct:
        category = "not_a_fit"
    elif not has_person(p) and not has_stakeholders:
        category = "contact_not_found"
    elif not has_person(p):
        category = "wrong_contact"  # company fits, no usable contact, stakeholders attached
    elif pfit.pct < t.person_fit_min_pct:
        category = "wrong_contact"
    elif total >= t.ideal_min_total and p.role_type == "buyer":
        category = "ideal"
    elif total >= t.good_min_total:
        category = "good"
    else:
        category = "wrong_contact"

    return Score(
        total=total,
        company_fit=cfit,
        person_fit=pfit,
        category=category,
        confidence=confidence(c, p),
        disqualified_reason=dq,
    )


def needs_stakeholders(p: PersonFacts | None, icp: ICP) -> bool:
    if not has_person(p):
        return True
    return person_fit(p, icp).pct < icp.thresholds.person_fit_min_pct or p.role_type != "buyer"
