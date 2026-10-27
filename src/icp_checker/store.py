"""Tiny local JSON-file store + disk cache. Good enough for a single-process prototype."""

import asyncio
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .config import get_settings
from .models import ICP, CompanyResearch, Lead, LeadList, Usage

_locks: dict[str, asyncio.Lock] = {}


def _lock(key: str) -> asyncio.Lock:
    return _locks.setdefault(key, asyncio.Lock())


def data_dir() -> Path:
    return get_settings().data_dir


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------- disk cache


def cache_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def cache_get(namespace: str, key: str, ttl_s: float | None) -> Any | None:
    hit = _read_json(data_dir() / "cache" / namespace / f"{key}.json")
    if hit is None:
        return None
    if ttl_s is not None and time.time() - hit["ts"] > ttl_s:
        return None
    return hit["value"]


def cache_set(namespace: str, key: str, value: Any) -> None:
    _write_json(data_dir() / "cache" / namespace / f"{key}.json", {"ts": time.time(), "value": value})


# ---------------------------------------------------------------- usage tracking

_usage_stack: ContextVar[tuple[Usage, ...]] = ContextVar("usage_stack", default=())


@contextmanager
def track_usage(usage: Usage):
    """Every Serper/LLM call inside this block is also counted on `usage` (nests)."""
    token = _usage_stack.set(_usage_stack.get() + (usage,))
    try:
        yield usage
    finally:
        _usage_stack.reset(token)


def record_usage(**deltas: int) -> None:
    for u in _usage_stack.get():
        for k, v in deltas.items():
            setattr(u, k, getattr(u, k) + v)


# ---------------------------------------------------------------- ICPs


def _icp_path(icp_id: str) -> Path:
    return data_dir() / "icps" / f"{icp_id}.json"


def load_icp_doc(icp_id: str) -> dict | None:
    return _read_json(_icp_path(icp_id))


def save_icp_draft(icp: ICP) -> ICP:
    doc = load_icp_doc(icp.id) or {"draft": None, "versions": []}
    icp = icp.model_copy(update={"status": "draft", "version": 0, "approved_by": None, "approved_at": None})
    doc["draft"] = icp.model_dump(mode="json")
    _write_json(_icp_path(icp.id), doc)
    return icp


def append_icp_version(icp: ICP) -> None:
    doc = load_icp_doc(icp.id) or {"draft": None, "versions": []}
    doc["versions"].append(icp.model_dump(mode="json"))
    _write_json(_icp_path(icp.id), doc)


def get_icp_draft(icp_id: str) -> ICP | None:
    doc = load_icp_doc(icp_id)
    return ICP.model_validate(doc["draft"]) if doc and doc.get("draft") else None


def get_icp_versions(icp_id: str) -> list[ICP]:
    doc = load_icp_doc(icp_id)
    return [ICP.model_validate(v) for v in doc["versions"]] if doc else []


def get_icp_version(icp_id: str, version: int | None = None) -> ICP | None:
    versions = get_icp_versions(icp_id)
    if not versions:
        return None
    if version is None:
        return versions[-1]
    return next((v for v in versions if v.version == version), None)


def list_icps() -> list[dict]:
    out = []
    for p in sorted((data_dir() / "icps").glob("*.json")):
        doc = _read_json(p)
        latest = doc["versions"][-1] if doc["versions"] else doc["draft"]
        out.append({
            "id": latest["id"], "name": latest["name"], "team": latest["team"],
            "latest_version": doc["versions"][-1]["version"] if doc["versions"] else None,
            "has_draft": doc["draft"] is not None,
        })
    return out


# ---------------------------------------------------------------- lists & leads


def _list_dir(list_id: str) -> Path:
    return data_dir() / "lists" / list_id


def save_list(lst: LeadList) -> None:
    _write_json(_list_dir(lst.id) / "list.json", lst.model_dump(mode="json"))


def get_list(list_id: str) -> LeadList | None:
    raw = _read_json(_list_dir(list_id) / "list.json")
    return LeadList.model_validate(raw) if raw else None


def all_lists() -> list[LeadList]:
    root = data_dir() / "lists"
    return [LeadList.model_validate(_read_json(p)) for p in sorted(root.glob("*/list.json"))] if root.exists() else []


def get_leads(list_id: str) -> list[Lead]:
    raw = _read_json(_list_dir(list_id) / "leads.json") or []
    return [Lead.model_validate(x) for x in raw]


def save_leads(list_id: str, leads: list[Lead]) -> None:
    _write_json(_list_dir(list_id) / "leads.json", [l.model_dump(mode="json") for l in leads])


async def upsert_lead(lead: Lead) -> None:
    """Write one lead as soon as it finishes so the UI sees rows appear progressively."""
    async with _lock(f"leads:{lead.list_id}"):
        leads = get_leads(lead.list_id)
        idx = {l.id: i for i, l in enumerate(leads)}
        if lead.id in idx:
            leads[idx[lead.id]] = lead
        else:
            leads.append(lead)
        save_leads(lead.list_id, leads)


def delete_list(list_id: str) -> bool:
    d = _list_dir(list_id)
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True


# ---------------------------------------------------------------- company research cache


def get_company_research(key: str, ttl_days: float | None = None) -> CompanyResearch | None:
    raw = _read_json(data_dir() / "companies" / f"{cache_key(key)[:24]}.json")
    if not raw:
        return None
    research = CompanyResearch.model_validate(raw)
    if ttl_days is not None and (time.time() - research.researched_at.timestamp()) > ttl_days * 86400:
        return None
    return research


def save_company_research(research: CompanyResearch) -> None:
    _write_json(data_dir() / "companies" / f"{cache_key(research.key)[:24]}.json", research.model_dump(mode="json"))
