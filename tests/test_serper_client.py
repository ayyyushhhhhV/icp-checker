import json

import httpx
import pytest

from icp_checker import serper as serper_mod
from icp_checker.models import Usage
from icp_checker.serper import Budget, SerperBudgetExceeded, SerperClient, SerperError, make_http_client
from icp_checker.store import track_usage


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(*a, **k):
        return None
    monkeypatch.setattr(serper_mod, "_backoff", instant)


def client_with(handler) -> SerperClient:
    return SerperClient(api_key="secret-key", http=make_http_client(transport=httpx.MockTransport(handler)))


async def test_search_request_shape_and_cache():
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(200, json={"organic": [{"title": "t", "link": "https://a.com"}]})

    c = client_with(handler)
    usage = Usage()
    with track_usage(usage):
        r1 = await c.search('"Acme" official website')
        r2 = await c.search('"Acme" official website')
    assert r1 == r2 and len(seen) == 1
    assert seen[0].url == "https://google.serper.dev/search"
    assert seen[0].headers["X-API-KEY"] == "secret-key"
    assert json.loads(seen[0].content) == {"q": '"Acme" official website', "num": 10}
    assert usage.serper_credits == 1 and usage.serper_cache_hits == 1


async def test_scrape_body():
    seen = []
    c = client_with(lambda req: seen.append(req) or httpx.Response(200, json={"markdown": "# hi"}))
    await c.scrape("https://acme.com")
    assert seen[0].url == "https://scrape.serper.dev"
    assert json.loads(seen[0].content) == {"url": "https://acme.com", "includeMarkdown": True}


async def test_retry_on_429_then_success():
    responses = iter([httpx.Response(429), httpx.Response(503), httpx.Response(200, json={"ok": 1})])
    c = client_with(lambda req: next(responses))
    assert await c.news("Acme") == {"ok": 1}


async def test_gives_up_after_max_retries():
    c = client_with(lambda req: httpx.Response(500))
    with pytest.raises(SerperError):
        await c.search("x")


async def test_budget_counts_only_real_calls():
    c = client_with(lambda req: httpx.Response(200, json={}))
    b = Budget(2)
    await c.search("a", budget=b)
    await c.search("a", budget=b)  # cached: free
    await c.search("b", budget=b)
    with pytest.raises(SerperBudgetExceeded):
        await c.search("c", budget=b)


async def test_offline_mode_uses_cache_only():
    c = client_with(lambda req: httpx.Response(200, json={"x": 1}))
    await c.search("cached")
    off = SerperClient(api_key="k", http=c.http, offline=True)
    assert await off.search("cached") == {"x": 1}
    with pytest.raises(SerperError):
        await off.search("not cached")
