#!/usr/bin/env python
"""Half-automatically build a run node from a training-queue job + log.

Given a job name (or a ``.job`` / ``.log`` path), this extracts the Hydra
overrides (model/loss/data/...) from the queued command and the final test
metric table from the log, then emits a run-node Markdown file. You still write
the 考察 body and link ``parents`` afterwards.

Usage:
    # by queue job name (searches .training_queue/{done,failed,jobs,logs})
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_from_run.py canon_both --task plcs

    # by explicit paths
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_from_run.py \
        --task plcs --job .training_queue/done/..._canon_both.job \
        --log .training_queue/logs/..._canon_both.log --write
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
from pathlib import Path

from kg_lib import dump_frontmatter, portable_path, queue_dir, repo_root
from kg_schema import PROVIDERS, iso_date
from kg_storage import save_node

sys.path.insert(0, str(repo_root()))
from src.utils.training_queue import (  # noqa: E402
    expired_completion,
    job_label,
    latest_named_job,
)

QUEUE_DIR = queue_dir()
METRIC_ROW_RE = re.compile(r"^[│|]\s*(test/\S+)\s*[│|]\s*([0-9.eE+-]+)\s*[│|]\s*$")
OVERRIDE_RE = re.compile(r"(?:^|\s)([\w.]+)=([^\s]+)")
# Hydra keys we surface into config (everything else is ignored as noise).
CONFIG_KEYS = ("model", "loss", "data")


def find_by_name(name: str) -> tuple[Path | None, Path | None]:
    job = None
    filename = latest_named_job(QUEUE_DIR, name)
    if filename is None:
        return None, None
    if expired_completion(QUEUE_DIR, filename) is not None:
        raise ValueError(f"queue history expired for {filename}; use kg_register.py with the retained repro bundle")
    for sub in ("running", "done", "failed", "cancelled", "jobs"):
        candidate = QUEUE_DIR / sub / filename
        if candidate.is_file():
            job = candidate
            break
    log = QUEUE_DIR / "logs" / f"{filename[:-4]}.log"
    return job, log if log.is_file() else None


def parse_job(job: Path) -> tuple[str, dict, str]:
    text = job.read_text(encoding="utf-8")
    name_m = re.search(r"^#\s*name:\s*(.+)$", text, re.MULTILINE)
    name = name_m.group(1).strip() if name_m else job_label(job.stem)
    cmd = next((ln for ln in text.splitlines() if "python" in ln and "-m " in ln), "")
    overrides = dict(OVERRIDE_RE.findall(cmd))
    config = {k: overrides[k] for k in CONFIG_KEYS if k in overrides}
    return name, config, cmd


def parse_log_metrics(log: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = METRIC_ROW_RE.match(line.strip())
        if m:
            key = m.group(1).replace("test/", "")
            with contextlib.suppress(ValueError):
                metrics[key] = round(float(m.group(2)), 6)
    return metrics


def status_from_job(job: Path | None) -> str:
    if job is None:
        return "planned"
    parent = job.parent.name
    return {"failed": "failed", "cancelled": "failed", "running": "running", "jobs": "planned"}.get(parent, "done")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", nargs="?", help="queue job name (e.g. canon_both)")
    p.add_argument("--job", type=Path)
    p.add_argument("--log", type=Path)
    p.add_argument("--task", required=True)
    p.add_argument("--date", help="actual experiment date, ISO YYYY-MM-DD")
    p.add_argument("--papers", nargs="*", default=[])
    p.add_argument("--id", help="node id (default: run-<name>)")
    p.add_argument("--issue", type=int)
    p.add_argument("--provider", choices=sorted(PROVIDERS))
    p.add_argument("--write", action="store_true", help="write to knowledge/nodes/ (else print)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.date and not iso_date(args.date):
        p.error("date must be YYYY-MM-DD")
    if args.issue is not None and args.issue <= 0:
        p.error("issue must be positive")

    job, log = args.job, args.log
    for path in (job, log):
        if path is not None and not path.exists() and expired_completion(QUEUE_DIR, path.stem + ".job") is not None:
            p.error("queue history expired; use kg_register.py with the retained repro bundle")
    if args.name and not (job and log):
        try:
            fj, fl = find_by_name(args.name)
        except ValueError as exc:
            p.error(str(exc))
        job = job or fj
        log = log or fl
    if job is None and log is None:
        p.error("could not locate a job or log; pass a name or --job/--log")

    name = args.name or job_label(job.stem if job else log.stem)
    config: dict[str, str] = {}
    cmd = ""
    if job:
        name, config, cmd = parse_job(job)
    metrics = parse_log_metrics(log) if log and log.exists() else {}

    node_id = args.id or f"run-{name.replace('_', '-')}"
    meta = {
        "id": node_id,
        "task": args.task,
        "papers": args.papers,
        "type": "run",
        "title": name,
        "issue": args.issue,
        "provider": args.provider,
        "date": args.date,
        "status": status_from_job(job),
        "config": config or {"model": "", "loss": "", "data": ""},
        "metrics": metrics,
        "artifacts": {
            "log": portable_path(log) if log else "",
            "job": portable_path(job) if job else "",
            "output_dir": "",
        },
        "parents": [],
        "relations": [],
        "tags": [],
    }
    meta = {k: v for k, v in meta.items() if v is not None}

    body = (
        "## 考察 / Findings\n\n"
        f"<!-- run `{name}` の結果と考察を書く。 -->\n"
    )
    doc = f"---\n{dump_frontmatter(meta)}---\n\n{body}"

    if args.write:
        try:
            out = save_node(meta, body, args.force)
        except ValueError as exc:
            p.error(str(exc))
        print(f"created {out}  (metrics: {len(metrics)}, config keys: {list(config)})")
    else:
        print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
