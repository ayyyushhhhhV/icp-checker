"""CLI for quick runs without the UI.

  uv run icp-checker draft-icp offering.txt --team hr_talent -o icp.json
  uv run icp-checker run --icp icp.json --csv leads.csv -o scored.csv
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
from pathlib import Path

from . import store
from .icp_service import approve_icp, draft_icp
from .models import ICP, LeadList
from .normalize import propose_mapping
from .pipeline import export_csv, run_list, summary


def _load_approved(icp_path: Path) -> ICP:
    icp = ICP.model_validate_json(icp_path.read_text())
    store.save_icp_draft(icp)
    return approve_icp(icp.id, approved_by="cli")


async def _draft(args) -> None:
    icp = await draft_icp(Path(args.offering).read_text(), team=args.team)
    Path(args.out).write_text(icp.model_dump_json(indent=2))
    print(f"Draft ICP written to {args.out} — review/edit it, then pass it to `run`.")


async def _run(args) -> None:
    icp = _load_approved(Path(args.icp))
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames or [] if c]
        rows = [{c: (r.get(c) or "") for c in columns} for r in reader]
    if args.limit:
        rows = rows[: args.limit]
    lst = LeadList(filename=Path(args.csv).name, columns=columns, mapping=propose_mapping(columns), rows=rows)
    print("Column mapping:", json.dumps(lst.mapping))
    store.save_list(lst)
    await run_list(lst.id, icp)
    lst = store.get_list(lst.id)
    Path(args.out).write_text(export_csv(lst))
    print(f"List {lst.id}: {lst.status}{' — ' + lst.error if lst.error else ''}")
    print("Summary:", json.dumps(summary(lst.id)))
    print("Usage:", lst.usage.model_dump_json())
    print(f"Results written to {args.out}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpx2", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="icp-checker")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft-icp", help="draft an ICP from an offering description")
    d.add_argument("offering", help="text file describing the offering")
    d.add_argument("--team", default="default")
    d.add_argument("-o", "--out", default="icp.json")
    r = sub.add_parser("run", help="score a CSV against an ICP json")
    r.add_argument("--icp", required=True)
    r.add_argument("--csv", required=True)
    r.add_argument("--limit", type=int, default=0, help="only process the first N rows")
    r.add_argument("-o", "--out", default="scored.csv")
    args = p.parse_args()
    try:
        asyncio.run(_draft(args) if args.cmd == "draft-icp" else _run(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
