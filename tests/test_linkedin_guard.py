import httpx
import pytest

from icp_checker.serper import LinkedInFetchForbidden, SerperClient, assert_not_linkedin, is_linkedin_url, make_http_client


@pytest.mark.parametrize("url", [
    "https://www.linkedin.com/in/jane-doe",
    "https://linkedin.com/company/acme",
    "http://in.linkedin.com/in/someone",
    "https://uk.linkedin.com/in/x?trk=1",
    "HTTPS://WWW.LINKEDIN.COM/in/x",
    "linkedin.com/in/no-scheme",
    "https://www.linkedin.com./in/trailing-dot",
])
def test_linkedin_urls_rejected(url):
    assert is_linkedin_url(url)
    with pytest.raises(LinkedInFetchForbidden):
        assert_not_linkedin(url)


@pytest.mark.parametrize("url", [
    "https://example.com", "https://notlinkedin.com/in/x", "https://linkedin.com.evil.io/x",
    "https://example.com/?ref=linkedin.com",
])
def test_other_urls_allowed(url):
    assert not is_linkedin_url(url)
    assert_not_linkedin(url)


async def test_scrape_refuses_linkedin_before_any_network_call():
    calls = []
    transport = httpx.MockTransport(lambda req: calls.append(req) or httpx.Response(200, json={}))
    client = SerperClient(api_key="k", http=make_http_client(transport=transport))
    with pytest.raises(LinkedInFetchForbidden):
        await client.scrape("https://www.linkedin.com/in/jane-doe")
    assert calls == []


async def test_http_client_hook_blocks_direct_linkedin_requests():
    transport = httpx.MockTransport(lambda req: httpx.Response(200))
    async with make_http_client(transport=transport) as http:
        with pytest.raises(LinkedInFetchForbidden):
            await http.get("https://www.linkedin.com/in/jane-doe")
