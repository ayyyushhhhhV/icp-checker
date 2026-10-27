import pytest

from icp_checker.config import get_settings
from icp_checker.models import ICP, CompanyCriteria, NeedSignal, Offering, RoleCriteria, TargetRole


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Every test gets its own data dir and no real credentials."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT_SMALL"):
        monkeypatch.setenv(k, "")
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)  # so a developer's .env is not picked up
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def icp() -> ICP:
    return ICP(
        id="icp-1", team="hr_talent", name="AI Recruiter", version=1, status="approved",
        offering=Offering(product="AI recruiting assistant"),
        company_criteria=CompanyCriteria(
            industries=["Retail", "Logistics"], excluded_industries=["Staffing agencies"],
            size_min_employees=200, size_max_employees=10000,
            locations=["India", "UAE"], excluded_locations=["China"],
            need_signals=[
                NeedSignal(signal="high-volume hiring", description="20+ open roles"),
                NeedSignal(signal="recent expansion", description="new offices or stores"),
            ],
        ),
        role_criteria=RoleCriteria(
            target_roles=[
                TargetRole(title_examples=["Head of Talent Acquisition", "TA Director"], function="talent_acquisition",
                           seniority_min="director", role_type="buyer"),
                TargetRole(title_examples=["Talent Partner Lead"], function="talent_acquisition",
                           seniority_min="manager", role_type="influencer"),
            ],
            deprioritized_functions=["finance"],
        ),
    )
