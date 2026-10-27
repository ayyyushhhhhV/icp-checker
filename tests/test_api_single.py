from fastapi.testclient import TestClient

import icp_checker.api as api
from icp_checker import store
from icp_checker.pipeline import lead_inputs


def test_single_lead_creates_one_row_list(monkeypatch, icp):
    started = []
    monkeypatch.setattr(api, "start_job", lambda list_id, icp: started.append((list_id, icp.version)))
    store.save_icp_draft(icp)
    from icp_checker.icp_service import approve_icp
    v = approve_icp(icp.id)
    c = TestClient(api.app)

    assert c.post("/leads/single", json={"icp_id": icp.id}).status_code == 400
    assert c.post("/leads/single", json={"icp_id": icp.id, "linkedin_url": "https://linkedin.com/company/x"}).status_code == 400
    assert c.post("/leads/single", json={"icp_id": "nope", "company": "Acme"}).status_code == 400

    r = c.post("/leads/single", json={"icp_id": icp.id, "linkedin_url": "https://in.linkedin.com/in/jane-doe", "company": "Acme Retail"})
    assert r.status_code == 200
    lst = store.get_list(r.json()["list_id"])
    [(_, inp)] = lead_inputs(lst)
    assert inp.linkedin_url == "https://in.linkedin.com/in/jane-doe" and inp.company == "Acme Retail"
    assert started == [(lst.id, v.version)]
