"""Core domain schemas: ICP, Company, Lead, Score, Stakeholder, Evidence."""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

SENIORITY_ORDER = ["intern", "entry", "senior_ic", "manager", "director", "vp", "c_level", "owner"]
Seniority = Literal["intern", "entry", "senior_ic", "manager", "director", "vp", "c_level", "owner"]
RoleType = Literal["buyer", "influencer", "none"]
Category = Literal["ideal", "good", "wrong_contact", "contact_not_found", "not_a_fit"]
Confidence = Literal["high", "medium", "low"]
SourceType = Literal["website", "careers", "news", "search_snippet", "rep_capture", "list"]


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class Evidence(BaseModel):
    fact: str
    source_url: str | None = None
    source_type: SourceType
    retrieved_at: datetime = Field(default_factory=now)


# ---------------------------------------------------------------- ICP


class Offering(BaseModel):
    product: str
    problems_solved: list[str] = []
    outcomes: list[str] = []


class NeedSignal(BaseModel):
    signal: str
    description: str = ""


class CompanyCriteria(BaseModel):
    industries: list[str] = []
    excluded_industries: list[str] = []
    size_min_employees: int | None = None
    size_max_employees: int | None = None
    locations: list[str] = []
    excluded_locations: list[str] = []
    need_signals: list[NeedSignal] = []


class TargetRole(BaseModel):
    title_examples: list[str]
    function: str
    seniority_min: Seniority
    role_type: Literal["buyer", "influencer"]


class RoleCriteria(BaseModel):
    target_roles: list[TargetRole] = []
    deprioritized_functions: list[str] = []
    excluded_seniority: list[Seniority] = ["intern", "entry"]


class Weights(BaseModel):
    industry: int = 20
    size: int = 15
    location: int = 10
    need_signals: int = 15
    function: int = 15
    seniority: int = 15
    role_type: int = 10

    @model_validator(mode="after")
    def _sum_100(self):
        total = sum(self.model_dump().values())
        if total != 100:
            raise ValueError(f"weights must sum to 100 (got {total})")
        return self

    @property
    def company(self) -> int:
        return self.industry + self.size + self.location + self.need_signals

    @property
    def person(self) -> int:
        return self.function + self.seniority + self.role_type


class Thresholds(BaseModel):
    company_fit_min_pct: int = 50
    ideal_min_total: int = 75
    good_min_total: int = 55
    person_fit_min_pct: int = 40


class ICP(BaseModel):
    id: str = Field(default_factory=new_id)
    team: str = "default"
    name: str
    version: int = 0  # 0 = unapproved draft; approvals create 1, 2, 3...
    status: Literal["draft", "approved"] = "draft"
    offering: Offering
    company_criteria: CompanyCriteria = CompanyCriteria()
    role_criteria: RoleCriteria = RoleCriteria()
    weights: Weights = Weights()
    thresholds: Thresholds = Thresholds()
    created_by: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None


# ---------------------------------------------------------------- Company


class SourceDoc(BaseModel):
    """Raw text gathered during research. Kept so facts can be re-extracted for a new ICP version
    without calling Serper again."""

    url: str
    source_type: SourceType
    title: str = ""
    text: str


class CompanyResearch(BaseModel):
    key: str
    name: str
    domain: str | None = None
    sources: list[SourceDoc] = []
    pages_scraped: int = 0
    notes: list[str] = []
    serper_calls: int = 0
    researched_at: datetime = Field(default_factory=now)


class CompanySignal(BaseModel):
    signal: str
    detail: str
    evidence: Evidence


class CompanyFacts(BaseModel):
    """ICP-aware normalized facts for one company (output of the extraction step)."""

    key: str
    name: str
    domain: str | None = None
    industry: str | None = None
    sub_industry: str | None = None
    industry_match: Literal["exact", "related", "none", "excluded", "unknown"] = "unknown"
    employee_estimate: int | None = None
    employee_range: str | None = None
    size_confidence: Confidence = "low"
    locations: list[str] = []
    matched_locations: list[str] = []
    excluded_location_hits: list[str] = []
    signals: list[CompanySignal] = []
    evidence: list[Evidence] = []
    thin_data: bool = False  # website could not be scraped / almost no text


# ---------------------------------------------------------------- Person / Score / Lead


class PersonFacts(BaseModel):
    name: str | None = None
    title: str | None = None
    title_source: Literal["list", "search_snippet", "rep_capture"] | None = None
    function: str | None = None
    seniority: Seniority | None = None
    role_type: RoleType = "none"
    matched_role_index: int | None = None
    verified: bool = False
    linkedin_url: str | None = None
    evidence: list[Evidence] = []


class FitPart(BaseModel):
    points: float
    max: int
    breakdown: dict[str, float]

    @property
    def pct(self) -> float:
        return 100.0 * self.points / self.max if self.max else 100.0


class Override(BaseModel):
    category: Category
    reason: str
    by: str | None = None
    at: datetime = Field(default_factory=now)


class Score(BaseModel):
    total: int
    company_fit: FitPart
    person_fit: FitPart
    category: Category
    confidence: Confidence
    disqualified_reason: str | None = None
    explanation: str = ""
    recommendation: str = ""
    overridden: bool = False
    override: Override | None = None


class StakeholderCandidate(BaseModel):
    """Raw candidate as found (before ICP normalization)."""

    name: str
    title: str | None = None
    linkedin_url: str | None = None  # from a search result link only; never fetched
    source: Literal["search_snippet", "company_website"]
    source_url: str | None = None


class Stakeholder(BaseModel):
    name: str
    title: str | None = None
    linkedin_url: str | None = None
    source: Literal["search_snippet", "company_website"]
    function: str | None = None
    seniority: Seniority | None = None
    role_type: RoleType = "none"
    score: Score
    relation_to_lead: str | None = None
    evidence: list[Evidence] = []


class LeadInput(BaseModel):
    name: str | None = None
    title: str | None = None
    company: str | None = None
    website: str | None = None
    linkedin_url: str | None = None
    email: str | None = None


class LinkedInProfile(BaseModel):
    """What one Google result (via Serper) says about a LinkedIn profile the list supplied. The page is never fetched."""

    slug: str
    url: str
    found: bool = False
    name: str | None = None
    title: str | None = None
    company: str | None = None
    result_title: str | None = None
    snippet: str | None = None


class Lead(BaseModel):
    id: str = Field(default_factory=new_id)
    list_id: str
    row_index: int
    input: LeadInput
    company_key: str | None = None
    company: CompanyFacts | None = None  # facts as extracted for icp_version
    linkedin_profile: LinkedInProfile | None = None  # set when the list row had a LinkedIn URL / ID
    raw_person: PersonFacts | None = None  # name/title/evidence before ICP normalization
    stakeholder_candidates: list[StakeholderCandidate] | None = None  # None = never searched
    person: PersonFacts | None = None
    score: Score | None = None
    stakeholders: list[Stakeholder] = []
    status: Literal["queued", "researching", "scored", "failed", "partial"] = "queued"
    error: str | None = None
    icp_id: str | None = None
    icp_version: int | None = None
    usage: dict[str, int] = {}


class Usage(BaseModel):
    serper_credits: int = 0
    serper_cache_hits: int = 0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        for k, v in other.model_dump().items():
            setattr(self, k, getattr(self, k) + v)


class LeadList(BaseModel):
    id: str = Field(default_factory=new_id)
    filename: str
    columns: list[str]
    mapping: dict[str, str | None]  # our field -> CSV column
    rows: list[dict[str, str]]
    status: Literal["uploaded", "running", "done", "failed"] = "uploaded"
    icp_id: str | None = None
    icp_version: int | None = None
    progress: dict[str, int] = {"total": 0, "done": 0}
    usage: Usage = Usage()
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
