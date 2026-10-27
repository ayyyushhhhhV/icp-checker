"""Serper client: search, news, scrape. The only way this service touches the web.

Hard rule: never fetch a linkedin.com page. LinkedIn data may only come from Google result
snippets that Serper returns for a search.
"""

import asyncio
import logging
import random
from urllib.parse import urlsplit

import httpx

from .config import get_settings
from .store import cache_get, cache_key, cache_set, record_usage

log = logging.getLogger(__name__)

SEARCH_URL = "https://google.serper.dev/search"
NEWS_URL = "https://google.serper.dev/news"
SCRAPE_URL = "https://scrape.serper.dev"


class LinkedInFetchForbidden(ValueError):
    """Raised when code tries to fetch a linkedin.com page."""


class SerperBudgetExceeded(RuntimeError):
    """Raised when a per-company Serper call cap is hit."""


class SerperError(RuntimeError):
    pass


def is_linkedin_url(url: str) -> bool:
    raw = url.strip()
    if "://" not in raw:
        raw = "https://" + raw
    host = (urlsplit(raw).hostname or "").lower().rstrip(".")
    return host == "linkedin.com" or host.endswith(".linkedin.com")


def assert_not_linkedin(url: str) -> None:
    if is_linkedin_url(url):
        raise LinkedInFetchForbidden(
            f"Refusing to fetch LinkedIn URL ({url}). LinkedIn data may only come from search snippets."
        )


async def _guard_request(request: httpx.Request) -> None:
    # Belt and braces: no request made through our HTTP clients may target linkedin.com directly.
    if is_linkedin_url(str(request.url)):
        raise LinkedInFetchForbidden(f"Refusing HTTP request to LinkedIn ({request.url}).")


def make_http_client(**kwargs) -> httpx.AsyncClient:
    hooks = kwargs.pop("event_hooks", {})
    hooks.setdefault("request", []).append(_guard_request)
    return httpx.AsyncClient(event_hooks=hooks, **kwargs)


class Budget:
    """Counts real (non-cached) Serper calls for one company."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def take(self) -> None:
        if self.used >= self.limit:
            raise SerperBudgetExceeded(f"Serper call cap ({self.limit}) reached for this company")
        self.used += 1


class SerperClient:
    def __init__(self, api_key: str | None = None, http: httpx.AsyncClient | None = None, offline: bool = False):
        s = get_settings()
        self.api_key = api_key if api_key is not None else s.serper_api_key.get_secret_value()
        self.http = http or make_http_client(timeout=s.http_timeout_s)
        self.offline = offline  # offline = cache only (used for re-scoring)
        self.ttl_s = s.serper_cache_ttl_days * 86400
        self.max_retries = s.http_max_retries

    async def aclose(self) -> None:
        await self.http.aclose()

    async def search(self, q: str, num: int = 10, budget: Budget | None = None, **extra) -> dict:
        return await self._post(SEARCH_URL, {"q": q, "num": num, **extra}, budget)

    async def news(self, q: str, num: int = 10, tbs: str | None = None, budget: Budget | None = None) -> dict:
        body = {"q": q, "num": num}
        if tbs:
            body["tbs"] = tbs
        return await self._post(NEWS_URL, body, budget)

    async def scrape(self, url: str, budget: Budget | None = None) -> dict:
        assert_not_linkedin(url)  # hard constraint — checked before cache or network
        return await self._post(SCRAPE_URL, {"url": url, "includeMarkdown": True}, budget)

    async def _post(self, endpoint: str, body: dict, budget: Budget | None) -> dict:
        key = cache_key(endpoint, body)
        cached = cache_get("serper", key, self.ttl_s)
        if cached is not None:
            record_usage(serper_cache_hits=1)
            return cached
        if self.offline:
            raise SerperError("offline mode: Serper call not in cache")
        if not self.api_key:
            raise SerperError("SERPER_API_KEY is not set")
        if budget:
            budget.take()

        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.http.post(
                    endpoint, json=body, headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"}
                )
            except httpx.TransportError as e:
                if attempt >= self.max_retries:
                    raise SerperError(f"network error: {e}") from e
                await _backoff(attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= self.max_retries:
                    raise SerperError(f"Serper {resp.status_code} after {attempt + 1} attempts")
                await _backoff(attempt, resp.headers.get("Retry-After"))
                continue
            record_usage(serper_credits=1)
            if resp.status_code >= 400:
                raise SerperError(f"Serper {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            cache_set("serper", key, data)
            log.info("serper call endpoint=%s q=%s", endpoint.rsplit("/", 1)[-1], body.get("q") or body.get("url"))
            return data
        raise SerperError("unreachable")


async def _backoff(attempt: int, retry_after: str | None = None) -> None:
    if retry_after and retry_after.isdigit():
        delay = float(retry_after)
    else:
        delay = min(2**attempt, 20) + random.uniform(0, 1)
    await asyncio.sleep(delay)
