#!/usr/bin/env python
"""List finished training-queue runs for a GitHub issue (experiment-result-pr step 3).

Filters queue jobs by the ``issue`` field recorded in each job's ``run.json`` —
not by job name — so a batch that mixes issues (e.g. ``i545_*`` jobs actually
tagged ``issue: 560``) is split correctly. For the target issue it shows each
job's status, whether its test-split predictions exist, whether it is already
registered under ``knowledge/``, and a ready-to-paste ``kg_register.py`` line
(provider taken from run.json). Jobs whose *name* implies the issue but whose
``run.json`` issue differs are flagged separately so they are not mis-registered.

Usage:
    .venv/bin/python .agents/skills/experiment-result-pr/scripts/discover_runs.py --issue 545
    .venv/bin/python .agents/skills/experiment-result-pr/scripts/discover_runs.py --issue 545 --json

Notes:
    - Run from the repo root; reads .training_queue/ and knowledge/ (honors KNOWLEDGE_DIR).
    - "done" jobs with predictions that are not yet registered are the ones to register.
    - Registration is detected by the job id appearing in a knowledge node's artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

KG_REGISTER = ".agents/skills/knowledge-control/scripts/kg_register.py"
NAME_ISSUE_RE = re.compile(r"\bi(\d+)")
LOG_JOBID_RE = re.compile(r"logs/(\d+_\d+_\S+?)\.log")


def repo_root() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()


ROOT = repo_root()
QUEUE = ROOT / ".training_queue"
KNOWLEDGE = Path(os.environ.get("KNOWLEDGE_DIR") or ROOT / "knowledge")


def job_status(jobid: str) -> str:
    for sub in ("running", "done", "failed", "jobs"):
        if (QUEUE / sub / f"{jobid}.job").exists():
            return "pending" if sub == "jobs" else sub
    return "?"


def registered_jobids() -> set[str]:
    """Job ids already promoted into the graph (a node records logs/<jobid>.log)."""
    nodes = KNOWLEDGE / "nodes"
    if not nodes.exists():
        return set()
    blob = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in nodes.glob("*.md"))
    return set(LOG_JOBID_RE.findall(blob))


def collect() -> list[dict]:
    repro = QUEUE / "repro"
    if not repro.exists():
        return []
    reg = registered_jobids()
    rows: list[dict] = []
    for rj in sorted(repro.glob("*/run.json")):
        jobid = rj.parent.name
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            data = {}
        name = data.get("name") or jobid
        m = NAME_ISSUE_RE.search(name)
        rows.append({
            "jobid": jobid,
            "name": name,
            "issue": str(data.get("issue") or "").strip(),
            "name_issue": m.group(1) if m else "",
            "provider": data.get("provider") or "",
            "status": job_status(jobid),
            "preds": (rj.parent / "predictions" / "metrics.json").exists(),
            "registered": jobid in reg,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--issue", required=True, help="target issue number")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()
    issue = str(args.issue).lstrip("#")

    rows = collect()
    mine = [r for r in rows if r["issue"] == issue]
    misfiled = [r for r in rows if r["name_issue"] == issue and r["issue"] != issue]

    if args.json:
        print(json.dumps({"issue": issue, "runs": mine, "misfiled": misfiled},
                         indent=2, ensure_ascii=False))
        return 0

    if not mine:
        print(f"no queue runs with run.json issue == {issue}")
    else:
        print(f"issue #{issue} — {len(mine)} queue run(s) (by run.json issue field)\n")
        print(f"  {'STATUS':7} {'PREDS':5} {'REG':3} {'PROVIDER':8} NAME")
        for r in sorted(mine, key=lambda r: r["jobid"]):
            print(f"  {r['status']:7} {('yes' if r['preds'] else 'no'):5} "
                  f"{('yes' if r['registered'] else '-'):3} {r['provider']:8} {r['name']}")
        todo = [r for r in mine if r["status"] == "done" and r["preds"] and not r["registered"]]
        if todo:
            print("\nto register (done + predictions + not yet registered):")
            for r in todo:
                prov = f" --provider {r['provider']}" if r["provider"] else ""
                print(f"  .venv/bin/python {KG_REGISTER} {r['name']} --issue {issue}{prov}")

    if misfiled:
        print(f"\n⚠ named 'i{issue}…' but run.json issue differs — do NOT register under #{issue}:")
        for r in sorted(misfiled, key=lambda r: r["jobid"]):
            print(f"  {r['name']:32} issue={r['issue'] or '?'}")
        print("  → register these under their own issue (discover_runs.py --issue <that-issue>)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
