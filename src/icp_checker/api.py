"""FastAPI app. Run: uv run uvicorn icp_checker.api:app --reload"""

import csv
import io
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import store
from .icp_service import approve_icp, draft_icp
from .llm import LLMError
from .models import ICP, Category, LeadList, Override
from .normalize import FIELD_ALIASES, linkedin_slug, propose_mapping
from .pipeline import export_csv, final_category, ranked, start_job, summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _name in ("httpx", "httpx2", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)  # keep per-request noise out of logs

app = FastAPI(title="ICP Checker (prototype)")
STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


# ---------------------------------------------------------------- ICPs


class DraftRequest(BaseModel):
    offering_text: str
    team: str = "default"
    answers: dict[str, str] = {}
    won_companies: list[str] = []


@app.post("/icps/draft", response_model=ICP)
async def icp_draft(req: DraftRequest):
    try:
        return await draft_icp(req.offering_text, req.team, req.answers, req.won_companies)
    except LLMError as e:
        raise HTTPException(502, str(e))


@app.post("/icps", response_model=ICP)
def icp_save(icp: ICP):
    return store.save_icp_draft(icp)


@app.get("/icps")
def icp_list():
    return store.list_icps()


@app.get("/icps/{icp_id}")
def icp_get(icp_id: str):
    doc = store.load_icp_doc(icp_id)
    if not doc:
        raise HTTPException(404, "ICP not found")
    return doc


@app.post("/icps/{icp_id}/approve", response_model=ICP)
def icp_approve(icp_id: str, approved_by: str | None = None):
    try:
        return approve_icp(icp_id, approved_by)
    except KeyError:
        raise HTTPException(404, "no draft to approve")


@app.get("/icps/{icp_id}/versions", response_model=list[ICP])
def icp_versions(icp_id: str):
    return store.get_icp_versions(icp_id)


# ---------------------------------------------------------------- lists


@app.post("/lists")
async def list_upload(file: UploadFile = File(...)):
    raw = (await file.read()).decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    columns = [c for c in (reader.fieldnames or []) if c]
    rows = [{c: (r.get(c) or "") for c in columns} for r in reader]
    if not columns or not rows:
        raise HTTPException(400, "CSV has no header or no rows")
    lst = LeadList(filename=file.filename or "upload.csv", columns=columns, mapping=propose_mapping(columns), rows=rows)
    store.save_list(lst)
    return {"id": lst.id, "columns": columns, "proposed_mapping": lst.mapping, "rows": len(rows), "preview": rows[:5]}


def _get_list(list_id: str) -> LeadList:
    lst = store.get_list(list_id)
    if not lst:
        raise HTTPException(404, "list not found")
    return lst


def _approved_icp(icp_id: str, version: int | None) -> ICP:
    icp = store.get_icp_version(icp_id, version)
    if not icp or icp.status != "approved":
        raise HTTPException(400, "ICP version not found or not approved (only approved ICPs can be used)")
    return icp


class StartRequest(BaseModel):
    icp_id: str
    version: int | None = None  # default: latest approved
    mapping: dict[str, str | None] | None = None  # confirm/override the proposed column mapping


@app.post("/lists/{list_id}/start")
async def list_start(list_id: str, req: StartRequest):
    lst = _get_list(list_id)
    if lst.status == "running":
        raise HTTPException(409, "list is already running")
    icp = _approved_icp(req.icp_id, req.version)
    if req.mapping is not None:
        bad = [c for c in req.mapping.values() if c and c not in lst.columns]
        if bad or set(req.mapping) - set(FIELD_ALIASES):
            raise HTTPException(400, f"invalid mapping (unknown columns: {bad})")
        lst.mapping = {f: req.mapping.get(f) for f in FIELD_ALIASES}
        store.save_list(lst)
    if not (lst.mapping.get("company") or lst.mapping.get("website") or lst.mapping.get("email")):
        raise HTTPException(400, "map at least one of company / website / email")
    start_job(list_id, icp)
    return {"status": "started", "icp_id": icp.id, "icp_version": icp.version}


@app.post("/lists/{list_id}/rescore")
async def list_rescore(list_id: str, req: StartRequest):
    """Re-score with another ICP version from stored facts. No new Serper calls."""
    lst = _get_list(list_id)
    if lst.status == "running":
        raise HTTPException(409, "list is already running")
    if not store.get_leads(list_id):
        raise HTTPException(400, "list has not been processed yet")
    icp = _approved_icp(req.icp_id, req.version)
    start_job(list_id, icp, rescore=True)
    return {"status": "rescoring", "icp_id": icp.id, "icp_version": icp.version}


@app.get("/lists")
def lists_all():
    return [{"id": l.id, "filename": l.filename, "status": l.status, "progress": l.progress, "created_at": l.created_at}
            for l in store.all_lists()]


@app.get("/lists/{list_id}")
def list_get(list_id: str):
    lst = _get_list(list_id)
    return {
        "id": lst.id, "filename": lst.filename, "status": lst.status, "error": lst.error,
        "icp_id": lst.icp_id, "icp_version": lst.icp_version, "columns": lst.columns, "mapping": lst.mapping,
        "progress": lst.progress, "usage": lst.usage, "summary": summary(list_id),
    }


@app.get("/lists/{list_id}/leads")
def list_leads(list_id: str, category: Category | None = None, sort: str = "score"):
    _get_list(list_id)
    leads = store.get_leads(list_id)
    if category:
        leads = [l for l in leads if final_category(l) == category]
    leads = ranked(leads) if sort == "score" else sorted(leads, key=lambda l: l.row_index)
    return [l.model_dump(mode="json") | {"final_category": final_category(l)} for l in leads]


@app.get("/lists/{list_id}/export.csv", response_class=PlainTextResponse)
def list_export(list_id: str):
    lst = _get_list(list_id)
    return PlainTextResponse(
        export_csv(lst), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="scored-{lst.filename}"'},
    )


@app.delete("/lists/{list_id}")
def list_delete(list_id: str):
    if not store.delete_list(list_id):
        raise HTTPException(404, "list not found")
    return {"deleted": list_id}


class SingleLeadRequest(BaseModel):
    icp_id: str
    version: int | None = None
    linkedin_url: str | None = None
    company: str | None = None
    name: str | None = None
    title: str | None = None
    website: str | None = None


SINGLE_COLUMNS = {"name": "Name", "title": "Title", "company": "Company", "website": "Website", "linkedin_url": "LinkedIn"}


@app.post("/leads/single")
async def lead_single(req: SingleLeadRequest):
    """Score one lead entered by hand (LinkedIn profile + company). Runs as a one-row list so results,
    export, override and re-score work the same way."""
    if not (req.linkedin_url or req.company or req.website):
        raise HTTPException(400, "give a LinkedIn profile, a company name or a website")
    if req.linkedin_url and not linkedin_slug(req.linkedin_url):
        raise HTTPException(400, "LinkedIn value must be a profile URL (linkedin.com/in/...) or a profile ID")
    icp = _approved_icp(req.icp_id, req.version)
    row = {col: (getattr(req, f) or "").strip() for f, col in SINGLE_COLUMNS.items()}
    label = req.name or (linkedin_slug(req.linkedin_url) if req.linkedin_url else None) or req.company or req.website
    lst = LeadList(
        filename=f"single: {label}", columns=list(SINGLE_COLUMNS.values()),
        mapping={f: SINGLE_COLUMNS.get(f) for f in FIELD_ALIASES}, rows=[row],
    )
    store.save_list(lst)
    start_job(lst.id, icp)
    return {"list_id": lst.id, "icp_id": icp.id, "icp_version": icp.version}


# ---------------------------------------------------------------- overrides


class OverrideRequest(BaseModel):
    category: Category
    reason: str
    by: str | None = None


@app.post("/lists/{list_id}/leads/{lead_id}/override")
async def lead_override(list_id: str, lead_id: str, req: OverrideRequest):
    if not req.reason.strip():
        raise HTTPException(400, "reason is required")
    lead = next((l for l in store.get_leads(list_id) if l.id == lead_id), None)
    if not lead or not lead.score:
        raise HTTPException(404, "scored lead not found")
    lead.score.overridden = True
    lead.score.override = Override(category=req.category, reason=req.reason.strip(), by=req.by)
    await store.upsert_lead(lead)
    return lead.model_dump(mode="json") | {"final_category": final_category(lead)}
