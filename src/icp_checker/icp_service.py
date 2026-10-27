"""ICP drafting (LLM) and versioning."""

from .config import get_settings
from .llm import LLM, LLMError
from .llm_schemas import ICPDraftOut
from .models import (
    ICP,
    CompanyCriteria,
    NeedSignal,
    Offering,
    RoleCriteria,
    TargetRole,
    Thresholds,
    Weights,
    now,
)
from .store import append_icp_version, get_icp_draft, get_icp_versions

WEIGHT_KEYS = ["industry", "size", "location", "need_signals", "function", "seniority", "role_type"]


def fix_weights(raw: dict[str, int]) -> Weights:
    """Scale model-proposed weights to sum to exactly 100 (largest-remainder rounding)."""
    vals = {k: max(0, int(raw.get(k, 0))) for k in WEIGHT_KEYS}
    total = sum(vals.values())
    if total == 0:
        return Weights()
    scaled = {k: v * 100 / total for k, v in vals.items()}
    out = {k: int(v) for k, v in scaled.items()}
    for k in sorted(scaled, key=lambda k: scaled[k] - out[k], reverse=True)[: 100 - sum(out.values())]:
        out[k] += 1
    return Weights(**out)


async def draft_icp(
    offering_text: str, team: str = "default", answers: dict[str, str] | None = None,
    won_companies: list[str] | None = None, llm: LLM | None = None,
) -> ICP:
    llm = llm or LLM()
    out = await llm.structured(
        "icp_draft", ICPDraftOut, team=team, offering_text=offering_text, answers=answers or {},
        won_companies=won_companies or [], functions=get_settings().functions,
    )
    if out is None:
        raise LLMError("could not produce a valid ICP draft")
    functions = set(get_settings().functions)
    return ICP(
        team=team,
        name=out.name,
        offering=Offering(product=out.product, problems_solved=out.problems_solved, outcomes=out.outcomes),
        company_criteria=CompanyCriteria(
            industries=out.industries, excluded_industries=out.excluded_industries,
            size_min_employees=out.size_min_employees, size_max_employees=out.size_max_employees,
            locations=out.locations, excluded_locations=out.excluded_locations,
            need_signals=[NeedSignal(signal=s.signal, description=s.description) for s in out.need_signals],
        ),
        role_criteria=RoleCriteria(
            target_roles=[
                TargetRole(**r.model_dump() | {"function": r.function if r.function in functions else "other"})
                for r in out.target_roles
            ],
            deprioritized_functions=[f for f in out.deprioritized_functions if f in functions],
            excluded_seniority=out.excluded_seniority,
        ),
        weights=fix_weights(out.weights.model_dump()),
        thresholds=Thresholds(**out.thresholds.model_dump()),
    )


def approve_icp(icp_id: str, approved_by: str | None = None) -> ICP:
    draft = get_icp_draft(icp_id)
    if draft is None:
        raise KeyError("no draft to approve")
    versions = get_icp_versions(icp_id)
    approved = draft.model_copy(update={
        "version": (versions[-1].version if versions else 0) + 1,
        "status": "approved", "approved_by": approved_by, "approved_at": now(),
    })
    append_icp_version(approved)  # immutable: versions are only ever appended
    return approved
