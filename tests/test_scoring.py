import pytest

from icp_checker.models import CompanyFacts, CompanySignal, Evidence, PersonFacts, Thresholds, Weights
from icp_checker.scoring import (
    company_fit,
    function_points,
    industry_points,
    location_points,
    need_signal_points,
    needs_stakeholders,
    person_fit,
    role_type_points,
    score,
    seniority_points,
    size_points,
)


def sig(name: str) -> CompanySignal:
    return CompanySignal(signal=name, detail="x", evidence=Evidence(fact="x", source_url="https://a.com", source_type="website"))


def company(**kw) -> CompanyFacts:
    base = dict(
        key="k", name="Acme", industry="Retail", industry_match="exact", employee_estimate=850,
        locations=["Bengaluru, India"], matched_locations=["India"],
        signals=[sig("high-volume hiring"), sig("recent expansion")],
    )
    return CompanyFacts(**(base | kw))


def person(**kw) -> PersonFacts:
    base = dict(name="Jane", title="Head of TA", title_source="list", function="talent_acquisition",
                seniority="director", role_type="buyer", matched_role_index=0)
    return PersonFacts(**(base | kw))


# ---------------------------------------------------------------- criteria


@pytest.mark.parametrize("match,points,dq", [("exact", 20, False), ("related", 10, False), ("none", 0, False),
                                             ("unknown", 0, False), ("excluded", 0, True)])
def test_industry(icp, match, points, dq):
    p, reason = industry_points(company(industry_match=match), icp)
    assert p == points and bool(reason) == dq


@pytest.mark.parametrize("n,points", [(850, 15), (200, 15), (10000, 15), (160, 7.5), (12000, 7.5),
                                      (149, 0), (12600, 0), (None, 0)])
def test_size(icp, n, points):
    assert size_points(company(employee_estimate=n), icp) == points


def test_size_unconstrained(icp):
    icp.company_criteria.size_min_employees = icp.company_criteria.size_max_employees = None
    assert size_points(company(employee_estimate=None), icp) == 15


def test_location(icp):
    assert location_points(company(), icp) == (10, None)
    assert location_points(company(matched_locations=[]), icp) == (0, None)
    assert location_points(company(matched_locations=["Germany"]), icp) == (0, None)  # not an ICP value
    p, reason = location_points(company(excluded_location_hits=["China"]), icp)
    assert p == 0 and "China" in reason


def test_need_signals(icp):
    assert need_signal_points(company(), icp) == 15
    assert need_signal_points(company(signals=[sig("high-volume hiring")]), icp) == 7.5
    assert need_signal_points(company(signals=[sig("made-up signal")]), icp) == 0
    assert need_signal_points(company(signals=[]), icp) == 0


def test_function(icp):
    assert function_points(person(), icp) == 15
    assert function_points(person(function="finance"), icp) == 0  # deprioritized
    assert function_points(person(function="marketing"), icp) == 3.75  # other = 25%
    assert function_points(person(function=None), icp) == 0


@pytest.mark.parametrize("seniority,points", [("director", 15), ("vp", 15), ("manager", 7.5), ("senior_ic", 0),
                                              ("intern", 0), ("entry", 0)])
def test_seniority(icp, seniority, points):
    assert seniority_points(person(seniority=seniority, matched_role_index=0), icp) == points


def test_seniority_uses_matched_role(icp):
    # Manager matched to the influencer role (seniority_min=manager) gets full seniority points
    assert seniority_points(person(seniority="manager", matched_role_index=1), icp) == 15


def test_role_type(icp):
    assert role_type_points(person(role_type="buyer"), icp) == 10
    assert role_type_points(person(role_type="influencer"), icp) == 6
    assert role_type_points(person(role_type="none"), icp) == 0


def test_person_fit_without_title_is_zero(icp):
    assert person_fit(None, icp).points == 0
    assert person_fit(person(title=None), icp).points == 0


# ---------------------------------------------------------------- categories


def test_ideal(icp):
    s = score(company(), person(), icp)
    assert s.total == 100 and s.category == "ideal" and s.company_fit.max == 60 and s.person_fit.max == 40


def test_good_when_influencer(icp):
    s = score(company(), person(role_type="influencer", seniority="manager", matched_role_index=1), icp)
    assert s.total == 96 and s.category == "good"  # ideal requires a buyer


def test_good_below_ideal(icp):
    s = score(company(signals=[], employee_estimate=None, matched_locations=[]), person(), icp)
    # company = 20, person = 40 -> 60 total, company fit 33% < 50% -> not a fit
    assert s.category == "not_a_fit"
    s = score(company(signals=[], employee_estimate=None), person(), icp)
    # company = 30/60 = 50% (passes), total 70 -> good
    assert s.total == 70 and s.category == "good"


def test_not_a_fit_excluded_industry(icp):
    s = score(company(industry_match="excluded"), person(), icp)
    assert s.category == "not_a_fit" and "Excluded industry" in s.disqualified_reason


def test_not_a_fit_excluded_location(icp):
    s = score(company(excluded_location_hits=["China"]), person(), icp)
    assert s.category == "not_a_fit" and "China" in s.disqualified_reason


def test_not_a_fit_low_company_fit(icp):
    s = score(company(industry_match="none", signals=[]), person(), icp)
    assert s.company_fit.pct < 50 and s.category == "not_a_fit" and s.disqualified_reason is None


def test_contact_not_found(icp):
    assert score(company(), None, icp, has_stakeholders=False).category == "contact_not_found"
    assert score(company(), None, icp, has_stakeholders=True).category == "wrong_contact"


def test_wrong_contact_low_person_fit(icp):
    s = score(company(), person(function="finance", seniority="manager", role_type="none", matched_role_index=None), icp)
    assert s.person_fit.pct < 40 and s.category == "wrong_contact"


def test_wrong_contact_low_total(icp):
    icp.thresholds = Thresholds(company_fit_min_pct=10, ideal_min_total=90, good_min_total=80, person_fit_min_pct=40)
    s = score(company(industry_match="none", signals=[]), person(), icp)
    assert s.total == 65 and s.category == "wrong_contact"


def test_confidence(icp):
    assert score(company(), person(), icp).confidence == "medium"
    assert score(company(), person(title_source="rep_capture"), icp).confidence == "high"
    assert score(company(thin_data=True), person(), icp).confidence == "low"
    assert score(company(employee_estimate=None), person(), icp).confidence == "low"


def test_deterministic(icp):
    a = score(company(), person(), icp).model_dump()
    b = score(company(), person(), icp).model_dump()
    assert a == b


def test_custom_weights(icp):
    icp.weights = Weights(industry=40, size=10, location=10, need_signals=0, function=20, seniority=10, role_type=10)
    s = score(company(industry_match="related"), person(), icp)
    assert s.company_fit.breakdown["industry"] == 20 and s.company_fit.max == 60


def test_weights_must_sum_to_100():
    with pytest.raises(ValueError):
        Weights(industry=50)


def test_needs_stakeholders(icp):
    assert needs_stakeholders(None, icp)
    assert not needs_stakeholders(person(), icp)
    assert needs_stakeholders(person(role_type="influencer", seniority="manager", matched_role_index=1), icp)
    assert needs_stakeholders(person(function="finance", role_type="none"), icp)


def test_company_fit_breakdown_keys(icp):
    part, _ = company_fit(company(), icp)
    assert set(part.breakdown) == {"industry", "size", "location", "need_signals"}
