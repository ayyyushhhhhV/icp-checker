from types import SimpleNamespace

import pytest

from icp_checker.config import get_settings
from icp_checker.llm import LLM
from icp_checker.llm_schemas import ExplainOut
from icp_checker.models import Usage
from icp_checker.store import track_usage


class FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


def fake_llm(contents, monkeypatch, small_deployment=""):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "main-dep")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_SMALL", small_deployment)
    get_settings.cache_clear()
    comp = FakeCompletions(contents)
    return LLM(client=SimpleNamespace(chat=SimpleNamespace(completions=comp))), comp


VARS = dict(company_name="Acme", lead_name="Jane", lead_title="HR", score={}, facts={}, evidence=[], stakeholders=[],
            search_titles=["Head of TA"])


@pytest.fixture
def explain_vars(icp):
    from icp_checker.models import CompanyFacts, FitPart, Score
    sc = Score(total=80, company_fit=FitPart(points=50, max=60, breakdown={}), person_fit=FitPart(points=30, max=40, breakdown={}),
               category="good", confidence="medium")
    return VARS | {"score": sc, "facts": CompanyFacts(key="k", name="Acme")}


async def test_strict_schema_and_success(monkeypatch, explain_vars):
    llm, comp = fake_llm(['{"explanation": "e", "recommendation": "r"}'], monkeypatch)
    usage = Usage()
    with track_usage(usage):
        out = await llm.structured("explain", ExplainOut, **explain_vars)
    assert out == ExplainOut(explanation="e", recommendation="r")
    rf = comp.calls[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["additionalProperties"] is False
    assert comp.calls[0]["model"] == "main-dep"
    assert usage.llm_calls == 1 and usage.prompt_tokens == 100


async def test_retry_once_with_error_then_success(monkeypatch, explain_vars):
    llm, comp = fake_llm(['{"explanation": "e"}', '{"explanation": "e", "recommendation": "r"}'], monkeypatch)
    out = await llm.structured("explain", ExplainOut, **explain_vars)
    assert out.recommendation == "r"
    assert len(comp.calls) == 2
    assert "failed validation" in comp.calls[1]["messages"][-1]["content"]


async def test_gives_up_returns_none(monkeypatch, explain_vars):
    llm, comp = fake_llm(["not json", '{"nope": 1}'], monkeypatch)
    assert await llm.structured("explain", ExplainOut, **explain_vars) is None
    assert len(comp.calls) == 2


async def test_small_falls_back_to_main_and_cache(monkeypatch, explain_vars):
    llm, comp = fake_llm(['{"explanation": "e", "recommendation": "r"}'], monkeypatch)
    await llm.structured("explain", ExplainOut, small=True, **explain_vars)
    assert comp.calls[0]["model"] == "main-dep"
    usage = Usage()
    with track_usage(usage):
        await llm.structured("explain", ExplainOut, small=True, **explain_vars)  # served from cache
    assert len(comp.calls) == 1 and usage.llm_cache_hits == 1


async def test_small_deployment_used_when_set(monkeypatch, explain_vars):
    llm, comp = fake_llm(['{"explanation": "e", "recommendation": "r"}'], monkeypatch, small_deployment="mini")
    await llm.structured("explain", ExplainOut, small=True, **explain_vars)
    assert comp.calls[0]["model"] == "mini"
