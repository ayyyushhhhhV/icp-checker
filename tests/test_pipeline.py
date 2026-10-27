"""End-to-end pipeline run with mocked Serper (httpx MockTransport) and a fake Azure LLM. No live calls."""

import json
from pathlib import Path

import httpx
import pytest

from icp_checker import store
from icp_checker.llm import LLM
from icp_checker.llm_schemas import (
    CompanyExtractOut,
    ExplainOut,
    SignalHit,
    SourcedText,
    TitleNorm,
    TitleNormalizeOut,
)
from icp_checker.models import LeadList
from icp_checker.normalize import propose_mapping
from icp_checker.pipeline import export_csv, run_list, summary
from icp_checker.serper import SerperClient, make_http_client

FIX = Path(__file__).parent / "fixtures"
PAGES = json.loads((FIX / "serper_pages.json").read_text())
SEARCH = json.loads((FIX / "serper_search.json").read_text())


class SerperMock:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        self.calls.append((str(req.url), body))
        if req.url.host == "scrape.serper.dev":
            return httpx.Response(200, json=PAGES.get(body["url"], {"markdown": ""}))
        if req.url.path == "/news":
            return httpx.Response(200, json={"news": []})
        q = body["q"]
        if q.startswith("site:linkedin.com/in (") and "Acme Retail" in q:
            return httpx.Response(200, json=SEARCH["stakeholders:Acme Retail"])
        return httpx.Response(200, json=SEARCH.get(q, {"organic": []}))


TITLES = {
    "Head of Talent Acquisition": TitleNorm(index=0, function="talent_acquisition", seniority="director", role_type="buyer", matched_role_index=0),
    "TA Director": TitleNorm(index=0, function="talent_acquisition", seniority="director", role_type="buyer", matched_role_index=0),
    "CFO": TitleNorm(index=0, function="finance", seniority="c_level", role_type="none", matched_role_index=None),
    "HR Manager": TitleNorm(index=0, function="hr", seniority="manager", role_type="none", matched_role_index=None),
}


class FakeLLM(LLM):
    def __init__(self):
        self.calls: list[str] = []

    async def structured(self, prompt_name, output, small=False, **v):
        self.calls.append(prompt_name)
        if prompt_name == "company_extract":
            if v["company_name"].startswith("Acme"):
                return CompanyExtractOut(
                    industry=SourcedText(value="Retail", source_url="https://acme.com"), sub_industry=None,
                    industry_match="exact", industry_match_reason=None,
                    employee_estimate=4000, employee_range="1001-5000", employee_source_url="https://acme.com/about",
                    size_confidence="high",
                    locations=[SourcedText(value="Bengaluru, India", source_url="https://acme.com/about"),
                               SourcedText(value="Dubai, UAE", source_url="https://invented.example")],
                    matched_icp_locations=["India", "UAE"], matched_excluded_locations=[],
                    signals=[SignalHit(signal_index=0, detail="140 store roles", source_url="https://acme.com/careers"),
                             SignalHit(signal_index=1, detail="40 new stores", source_url="https://acme.com")],
                )
            return CompanyExtractOut(
                industry=SourcedText(value="Staffing agency", source_url="https://globex.com"), sub_industry=None,
                industry_match="excluded", industry_match_reason="staffing", employee_estimate=None, employee_range=None,
                employee_source_url=None, size_confidence="low",
                locations=[SourcedText(value="China", source_url="https://globex.com")],
                matched_icp_locations=[], matched_excluded_locations=["China"], signals=[],
            )
        if prompt_name == "title_normalize":
            return TitleNormalizeOut(items=[
                TITLES[t].model_copy(update={"index": i}) for i, t in enumerate(v["titles"]) if t in TITLES
            ])
        if prompt_name == "explain":
            return ExplainOut(explanation=f"{v['score'].category} because rules.", recommendation="Do the thing")
        return None


CSV_ROWS = [
    {"Full Name": "Jane Doe", "Designation": "Head of Talent Acquisition", "Company": "Acme Retail Pvt Ltd", "Website": "acme.com"},
    {"Full Name": "Bob Banker", "Designation": "CFO", "Company": "Acme Retail", "Website": "https://www.acme.com"},
    {"Full Name": "", "Designation": "", "Company": "Acme Retail", "Website": "acme.com"},
    {"Full Name": "Zed Zee", "Designation": "HR Manager", "Company": "Globex Staffing", "Website": "globex.com"},
    {"Full Name": "Jane Doe", "Designation": "Head of Talent Acquisition", "Company": "Acme Retail", "Website": "acme.com"},  # dup
]


@pytest.fixture
async def ran(icp):
    cols = list(CSV_ROWS[0])
    lst = LeadList(filename="t.csv", columns=cols, mapping=propose_mapping(cols), rows=CSV_ROWS)
    store.save_list(lst)
    mock = SerperMock()
    serper = SerperClient(api_key="k", http=make_http_client(transport=httpx.MockTransport(mock)))
    llm = FakeLLM()
    await run_list(lst.id, icp, serper=serper, llm=llm)
    leads = {l.row_index: l for l in store.get_leads(lst.id)}
    return lst, leads, mock, llm


async def test_categories(ran):
    lst, leads, _, _ = ran
    assert sorted(leads) == [0, 1, 2, 3]  # duplicate row 4 dropped
    assert leads[0].score.category == "ideal" and leads[0].score.total == 100
    assert leads[0].person.verified and leads[0].person.linkedin_url.startswith("https://in.linkedin.com/in/")
    assert leads[1].score.category == "wrong_contact"
    assert [s.name for s in leads[1].stakeholders] == ["Rahul Sharma"]
    assert leads[2].score.category == "wrong_contact" and leads[2].stakeholders  # company only -> stakeholder found
    assert leads[3].score.category == "not_a_fit" and "Excluded" in leads[3].score.disqualified_reason
    assert all(l.score.explanation for l in leads.values())
    assert store.get_list(lst.id).status == "done"
    assert summary(lst.id)["counts"] == {"ideal": 1, "wrong_contact": 2, "not_a_fit": 1}


async def test_company_researched_once_and_never_scrapes_linkedin(ran):
    _, leads, mock, llm = ran
    scraped = [b["url"] for u, b in mock.calls if "scrape" in u]
    assert sorted(scraped) == ["https://acme.com", "https://acme.com/about", "https://acme.com/careers", "https://globex.com"]
    assert not any("linkedin.com" in u for u in scraped)
    assert llm.calls.count("company_extract") == 2


async def test_not_a_fit_skips_person_and_stakeholder_search(ran):
    _, _, mock, _ = ran
    queries = [b.get("q", "") for u, b in mock.calls]
    assert not any("site:linkedin.com" in q and "Globex" in q for q in queries)
    stakeholder_searches = [q for q in queries if q.startswith("site:linkedin.com/in (")]
    assert len(stakeholder_searches) == 1  # once per company, shared by its leads


async def test_unsourced_facts_dropped(ran):
    _, leads, _, _ = ran
    assert "Dubai, UAE" not in leads[0].company.locations  # source_url was not one we provided
    assert all(e.source_url for e in leads[0].company.evidence)


async def test_rescore_offline_with_new_version(ran, icp):
    lst, _, mock, _ = ran
    n_calls = len(mock.calls)
    stricter = icp.model_copy(update={"version": 2}, deep=True)
    stricter.thresholds.ideal_min_total = 101  # nothing can be ideal now
    await run_list(lst.id, stricter, rescore=True, llm=FakeLLM(),
                   serper=SerperClient(api_key="k", offline=True, http=make_http_client(transport=httpx.MockTransport(mock))))
    leads = {l.row_index: l for l in store.get_leads(lst.id)}
    assert len(mock.calls) == n_calls  # no new Serper calls
    assert leads[0].score.category == "good" and leads[0].icp_version == 2
    assert leads[1].stakeholders  # rebuilt from stored candidates


async def test_export_csv(ran):
    lst, _, _, _ = ran
    out = export_csv(store.get_list(lst.id)).splitlines()
    assert out[0].startswith("Full Name,Designation,Company,Website,Category,Score")
    assert out[1].startswith("Jane Doe") and ",ideal,100," in out[1]


async def test_linkedin_only_rows(icp):
    """A row can be just a LinkedIn URL/ID: name, title and company come from the search result, page never fetched."""
    rows = [
        {"Name": "", "Company": "", "LinkedIn": "https://in.linkedin.com/in/rahul-sharma-ta/"},
        {"Name": "", "Company": "", "LinkedIn": "nobody-here"},
    ]
    lst = LeadList(filename="li.csv", columns=list(rows[0]), mapping=propose_mapping(list(rows[0])), rows=rows)
    assert lst.mapping["linkedin_url"] == "LinkedIn"
    store.save_list(lst)
    mock = SerperMock()
    serper = SerperClient(api_key="k", http=make_http_client(transport=httpx.MockTransport(mock)))
    await run_list(lst.id, icp, serper=serper, llm=FakeLLM())
    leads = {l.row_index: l for l in store.get_leads(lst.id)}

    rahul = leads[0]
    assert rahul.linkedin_profile.found and rahul.linkedin_profile.company == "Acme Retail"
    assert rahul.person.name == "Rahul Sharma" and rahul.person.title == "TA Director"
    assert rahul.person.verified and rahul.person.title_source == "search_snippet"
    assert rahul.score.category == "ideal"
    queries = [b.get("q", "") for u, b in mock.calls]
    assert not any(q.startswith('site:linkedin.com/in "') for q in queries)  # no extra name search needed
    assert not any("linkedin.com" in b.get("url", "") for u, b in mock.calls)  # never scraped

    assert leads[1].status == "failed" and "not found" in leads[1].error
    assert store.get_list(lst.id).progress == {"total": 2, "done": 2}
