"""Pydantic output models, one per prompt template. Strict-schema friendly: every field is required
(nullable where a value may be unknown) and there are no non-null defaults."""

from typing import Literal

from pydantic import BaseModel

from .models import RoleType, Seniority


# ---- icp_draft
class DraftNeedSignal(BaseModel):
    signal: str
    description: str


class DraftTargetRole(BaseModel):
    title_examples: list[str]
    function: str
    seniority_min: Seniority
    role_type: Literal["buyer", "influencer"]


class DraftWeights(BaseModel):
    industry: int
    size: int
    location: int
    need_signals: int
    function: int
    seniority: int
    role_type: int


class DraftThresholds(BaseModel):
    company_fit_min_pct: int
    ideal_min_total: int
    good_min_total: int
    person_fit_min_pct: int


class ICPDraftOut(BaseModel):
    name: str
    product: str
    problems_solved: list[str]
    outcomes: list[str]
    industries: list[str]
    excluded_industries: list[str]
    size_min_employees: int | None
    size_max_employees: int | None
    locations: list[str]
    excluded_locations: list[str]
    need_signals: list[DraftNeedSignal]
    target_roles: list[DraftTargetRole]
    deprioritized_functions: list[str]
    excluded_seniority: list[Seniority]
    weights: DraftWeights
    thresholds: DraftThresholds


# ---- company_extract
class SourcedText(BaseModel):
    value: str
    source_url: str


class SignalHit(BaseModel):
    signal_index: int
    detail: str
    source_url: str


class CompanyExtractOut(BaseModel):
    industry: SourcedText | None
    sub_industry: SourcedText | None
    industry_match: Literal["exact", "related", "none", "excluded", "unknown"]
    industry_match_reason: str | None
    employee_estimate: int | None
    employee_range: str | None
    employee_source_url: str | None
    size_confidence: Literal["high", "medium", "low"]
    locations: list[SourcedText]
    matched_icp_locations: list[str]
    matched_excluded_locations: list[str]
    signals: list[SignalHit]


# ---- domain_pick
class DomainPickOut(BaseModel):
    domain: str | None
    source_url: str | None


# ---- title_normalize
class TitleNorm(BaseModel):
    index: int
    function: str | None
    seniority: Seniority | None
    role_type: RoleType
    matched_role_index: int | None


class TitleNormalizeOut(BaseModel):
    items: list[TitleNorm]


# ---- snippet_parse
class ParsedPerson(BaseModel):
    name: str
    title: str | None
    company: str | None
    link: str


class SnippetParseOut(BaseModel):
    people: list[ParsedPerson]


# ---- team_page_extract
class TeamMember(BaseModel):
    name: str
    title: str
    source_url: str


class TeamPageOut(BaseModel):
    people: list[TeamMember]


# ---- explain
class ExplainOut(BaseModel):
    explanation: str
    recommendation: str


# ---- profile_snippet_parse
class ProfileSnippetOut(BaseModel):
    name: str | None
    title: str | None
    company: str | None
