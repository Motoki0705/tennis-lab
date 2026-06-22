#!/usr/bin/env python
"""Half-automatically build a run node from a training-queue job + log.

Given a job name (or a ``.job`` / ``.log`` path), this extracts the Hydra
overrides (model/loss/data/...) from the queued command and the final test
metric table from the log, then emits a run-node Markdown file. You still write
the 考察 body and link ``parents`` afterwards.

Usage:
    # by queue job name (searches .training_queue/{done,failed,jobs,logs})
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_from_run.py canon_both

    # by explicit paths
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_from_run.py \
        --job .training_queue/done/..._canon_both.job \
        --log .training_queue/logs/..._canon_both.log --write
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from kg_lib import dump_frontmatter, nodes_dir, repo_root

QUEUE_DIR = repo_root() / ".training_queue"
METRIC_ROW_RE = re.compile(r"^[│|]\s*(test/\S+)\s*[│|]\s*([0-9.eE+-]+)\s*[│|]\s*$")
OVERRIDE_RE = re.compile(r"(?:^|\s)([\w.]+)=([^\s]+)")
# Hydra keys we surface into config (everything else is ignored as noise).
CONFIG_KEYS = ("model", "loss", "data")


def find_by_name(name: str) -> tuple[Path | None, Path | None]:
    job = None
    for sub in ("running", "done", "failed", "jobs"):
        hits = sorted((QUEUE_DIR / sub).glob(f"*_{name}.job")) if (QUEUE_DIR / sub).exists() else []
        if hits:
            job = hits[-1]
            break
    log_hits = sorted((QUEUE_DIR / "logs").glob(f"*_{name}.log")) if (QUEUE_DIR / "logs").exists() else []
    log = log_hits[-1] if log_hits else None
    return job, log


def parse_job(job: Path) -> tuple[str, dict, str]:
    text = job.read_text(encoding="utf-8")
    name_m = re.search(r"^#\s*name:\s*(.+)$", text, re.MULTILINE)
    name = name_m.group(1).strip() if name_m else job.stem.split("_", 2)[-1]
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
            try:
                metrics[key] = round(float(m.group(2)), 6)
            except ValueError:
                pass
    return metrics


def status_from_job(job: Path | None) -> str:
    if job is None:
        return "planned"
    parent = job.parent.name
    return {"failed": "failed", "running": "running", "jobs": "planned"}.get(parent, "done")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", nargs="?", help="queue job name (e.g. canon_both)")
    p.add_argument("--job", type=Path)
    p.add_argument("--log", type=Path)
    p.add_argument("--id", help="node id (default: run-<name>)")
    p.add_argument("--issue", type=int)
    p.add_argument("--provider", default="claude")
    p.add_argument("--write", action="store_true", help="write to knowledge/nodes/ (else print)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    job, log = args.job, args.log
    if args.name and not (job and log):
        fj, fl = find_by_name(args.name)
        job = job or fj
        log = log or fl
    if job is None and log is None:
        p.error("could not locate a job or log; pass a name or --job/--log")

    name = args.name or (job.stem.split("_", 2)[-1] if job else log.stem.split("_", 2)[-1])
    config, cmd = {}, ""
    if job:
        name, config, cmd = parse_job(job)
    metrics = parse_log_metrics(log) if log and log.exists() else {}

    node_id = args.id or f"run-{name.replace('_', '-')}"
    meta = {
        "id": node_id,
        "type": "run",
        "title": name,
        "issue": args.issue,
        "provider": args.provider,
        "date": None,
        "status": status_from_job(job),
        "config": config or {"model": "", "loss": "", "data": ""},
        "metrics": metrics,
        "artifacts": {
            "log": str(log.relative_to(repo_root())) if log else "",
            "job": str(job.relative_to(repo_root())) if job else "",
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
        out = nodes_dir() / f"{node_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not args.force:
            p.error(f"{out} exists (use --force)")
        out.write_text(doc, encoding="utf-8")
        print(f"created {out}  (metrics: {len(metrics)}, config keys: {list(config)})")
    else:
        print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
