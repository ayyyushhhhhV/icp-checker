import pytest

from icp_checker.normalize import (
    company_matches,
    is_linkedin_profile_link,
    name_matches,
    normalize_company_name,
    normalize_domain,
    parse_linkedin_result_title,
    propose_mapping,
    search_company_name,
    company_from_snippet,
    linkedin_slug,
)


def test_mapping_common_headers():
    cols = ["Full Name", "Designation", "Organisation", "Company Website", "LinkedIn Profile URL", "Work Email"]
    m = propose_mapping(cols)
    assert m == {
        "name": "Full Name", "title": "Designation", "company": "Organisation",
        "linkedin_url": "LinkedIn Profile URL", "email": "Work Email", "website": "Company Website",
    }


def test_mapping_simple_and_missing():
    m = propose_mapping(["name", "Job Title", "Company Name", "domain"])
    assert m["name"] == "name" and m["title"] == "Job Title" and m["company"] == "Company Name"
    assert m["website"] == "domain" and m["linkedin_url"] is None and m["email"] is None


@pytest.mark.parametrize("raw,expected", [
    ("Acme Retail Pvt Ltd", "acme retail"),
    ("Acme Retail Private Limited", "acme retail"),
    ("ACME Retail, Inc.", "acme retail"),
    ("Tata Consultancy Services Limited", "tata consultancy services"),
    ("Landmark Group LLC", "landmark"),
    ("Café Coffee Day (CCD)", "cafe coffee day"),
    ("M&M Logistics", "m and m logistics"),
])
def test_company_name_normalization(raw, expected):
    assert normalize_company_name(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("https://www.Acme.com/about", "acme.com"), ("acme.co.in", "acme.co.in"), ("jane@acme.com", "acme.com"),
    ("", None), ("not a domain", None),
])
def test_domain_normalization(raw, expected):
    assert normalize_domain(raw) == expected


@pytest.mark.parametrize("title,expected", [
    ("Jane Doe - Head of Talent Acquisition - Acme Retail | LinkedIn",
     {"name": "Jane Doe", "title": "Head of Talent Acquisition", "company": "Acme Retail"}),
    ("Rahul Sharma – TA Director – Acme Retail Pvt Ltd - LinkedIn",
     {"name": "Rahul Sharma", "title": "TA Director", "company": "Acme Retail Pvt Ltd"}),
    ("Priya Nair - Talent Partner Lead at Acme Retail | LinkedIn",
     {"name": "Priya Nair", "title": "Talent Partner Lead", "company": "Acme Retail"}),
    ("John Smith - Acme Retail | LinkedIn", {"name": "John Smith", "title": None, "company": "Acme Retail"}),
    ("Devi Dash, SHRM-SCP, CPHR - |CHRO - Acme Retail | LinkedIn", {"name": "Devi Dash", "title": "CHRO", "company": "Acme Retail"}),
    ("\u200fShalini V\u200f - \u200fCHRO| VP HR - Acme Retail | LinkedIn", {"name": "Shalini V", "title": "CHRO| VP HR", "company": "Acme Retail"}),
    ("Mayank R - Group Chief Human Resources Officer at - Acme Retail",
     {"name": "Mayank R", "title": "Group Chief Human Resources Officer", "company": "Acme Retail"}),
])
def test_parse_linkedin_result_title(title, expected):
    assert parse_linkedin_result_title(title, company="Acme Retail") == expected


def test_parse_rejects_non_person_titles():
    assert parse_linkedin_result_title("Acme | LinkedIn") is None
    assert parse_linkedin_result_title("") is None


def test_search_company_name():
    assert search_company_name("Acme Retail Pvt Ltd") == "Acme Retail"
    assert search_company_name("Landmark Group LLC") == "Landmark Group"
    assert search_company_name("Tata Consultancy Services Limited") == "Tata Consultancy Services"
    assert search_company_name("Infosys") == "Infosys"


def test_fuzzy_matching():
    assert name_matches("Jane Doe", "Jane A. Doe")
    assert name_matches("Rahul Sharma", "Rahul Sharmaa")
    assert not name_matches("Jane Doe", "John Doe")
    assert company_matches("Acme Retail Pvt Ltd", "Head of TA at Acme Retail")
    assert company_matches("Acme Retail", "ACME RETAIL LIMITED")
    assert not company_matches("Acme Retail", "Globex Logistics")


def test_profile_link():
    assert is_linkedin_profile_link("https://in.linkedin.com/in/jane")
    assert not is_linkedin_profile_link("https://www.linkedin.com/company/acme")


@pytest.mark.parametrize("value,slug", [
    ("https://in.linkedin.com/in/merrilmohan", "merrilmohan"),
    ("https://www.linkedin.com/in/Jane-Doe-123/?originalSubdomain=in", "jane-doe-123"),
    ("linkedin.com/in/jane-doe-123/", "jane-doe-123"),
    ("/in/jane-doe-123", "jane-doe-123"),
    ("jane-doe-123", "jane-doe-123"),
    ("@janedoe", "janedoe"),
    ("https://www.linkedin.com/in/j%C3%A9r%C3%B4me-x", "jérôme-x"),
    ("https://www.linkedin.com/company/acme", None),
    ("Jane Doe", None),
    ("https://example.com/in/x", None),
    ("", None),
])
def test_linkedin_slug(value, slug):
    assert linkedin_slug(value) == slug


def test_company_from_snippet():
    assert company_from_snippet("Head of TA. Experience: Acme Retail · Education: IIM · Location: Pune") == "Acme Retail"
    assert company_from_snippet("No experience line here") is None
