#!/usr/bin/env python
"""Promote a finished training-queue run into the git-tracked knowledge graph.

This is the canonical entry point for the issue #533 workflow. Given a queue job
name (or an explicit ``--repro-dir``) it:

1. Copies the reproducibility bundle (``run.json`` / ``repro.sh`` /
   ``uncommitted.patch`` / ``git_status.txt``) and the test-split predictions
   (``pred_test.npz`` / ``metrics.json`` / ``diagnostic_metrics.json``) out of
   the gitignored
   ``.training_queue/`` staging area into git-tracked
   ``knowledge/runs/<run-id>/``.
2. Scaffolds a run node ``knowledge/nodes/<run-id>.md`` with frontmatter filled
   from ``run.json`` (provider / session / issue / commit / branch / remote /
   command), config from the command overrides, metrics from
   ``predictions/metrics.json`` (or the queue log), and ``artifacts`` pointing at
   the promoted bundle.

You still write the 考察 body and link ``parents`` / ``tags`` afterwards, then
run ``kg_validate.py``.

Usage:
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_register.py \
        i525_asym --issue 525 --provider claude
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_register.py \
        --repro-dir .training_queue/repro/<jobid> --id run-i525-asym --issue 525
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
from pathlib import Path

from kg_lib import dump_frontmatter, nodes_dir, repo_root

QUEUE_DIR = repo_root() / ".training_queue"
OVERRIDE_RE = re.compile(r"(?:^|\s)([\w.]+)=([^\s]+)")
CONFIG_KEYS = ("model", "loss", "data")
METRIC_ROW_RE = re.compile(r"^[│|]\s*(test/\S+)\s*[│|]\s*([0-9.eE+-]+)\s*[│|]\s*$")
BUNDLE_FILES = ("run.json", "repro.sh", "uncommitted.patch", "git_status.txt", "output_dir.txt")
PREDICTION_FILES = (
    "pred_test.npz",
    "metrics.json",
    "diagnostic_metrics.json",
)


def portable_output_dir(raw: str, run: dict) -> str:
    """Make the runner's output dir portable for ``artifacts.output_dir``.

    The runner records the Lightning ``version_N/checkpoints`` dir; the
    TensorBoard event files live in its parent ``version_N``. Drop a trailing
    ``checkpoints`` and store the path relative to the checkout that produced it,
    so kg_curves can resolve the event dir directly (no fingerprint scan)."""
    odir = Path(raw)
    if odir.name == "checkpoints":
        odir = odir.parent
    for base in (run.get("cwd"), str(repo_root())):
        if base:
            try:
                return str(odir.relative_to(base))
            except ValueError:
                continue
    return str(odir)


def find_repro_dir(name: str) -> Path | None:
    """Locate the repro dir for a queue job by name (jobid = job filename stem)."""
    for sub in ("running", "done", "failed", "jobs"):
        d = QUEUE_DIR / sub
        if d.exists():
            hits = sorted(d.glob(f"*_{name}.job"))
            if hits:
                return QUEUE_DIR / "repro" / hits[-1].stem
    repro_root = QUEUE_DIR / "repro"
    if repro_root.exists():
        hits = sorted(repro_root.glob(f"*_{name}"))
        if hits:
            return hits[-1]
    return None


def find_log(name: str) -> Path | None:
    d = QUEUE_DIR / "logs"
    hits = sorted(d.glob(f"*_{name}.log")) if d.exists() else []
    return hits[-1] if hits else None


def parse_metrics_from_log(log: Path | None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if log and log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = METRIC_ROW_RE.match(line.strip())
            if m:
                with contextlib.suppress(ValueError):
                    metrics[m.group(1).replace("test/", "")] = round(float(m.group(2)), 6)
    return metrics


def load_metrics(repro: Path, name: str) -> dict[str, float]:
    mj = repro / "predictions" / "metrics.json"
    if mj.exists():
        try:
            raw = json.loads(mj.read_text(encoding="utf-8"))
            return {k: round(float(v), 6) for k, v in raw.items()}
        except (ValueError, TypeError):
            pass
    return parse_metrics_from_log(find_log(name))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", nargs="?", help="queue job name (e.g. i525_asym)")
    p.add_argument("--repro-dir", type=Path, help="explicit .training_queue/repro/<jobid>")
    p.add_argument("--id", help="node id (default: run-<name>)")
    p.add_argument("--issue", type=int)
    p.add_argument("--provider")
    p.add_argument("--force", action="store_true", help="overwrite existing node / bundle")
    args = p.parse_args()

    repro = args.repro_dir
    if repro is None and args.name:
        repro = find_repro_dir(args.name)
    if repro is None or not repro.exists():
        p.error("could not locate a repro dir; pass a job name or --repro-dir")

    run = {}
    run_json = repro / "run.json"
    if run_json.exists():
        run = json.loads(run_json.read_text(encoding="utf-8"))

    name = args.name or run.get("name") or repro.name
    provider = args.provider or run.get("provider") or "claude"
    issue = args.issue
    if issue is None and str(run.get("issue", "")).isdigit():
        issue = int(run["issue"])
    cmd = run.get("command", "")
    overrides = dict(OVERRIDE_RE.findall(cmd))
    config = {k: overrides[k] for k in CONFIG_KEYS if k in overrides} or {
        "model": "", "loss": "", "data": ""
    }
    metrics = load_metrics(repro, name)

    node_id = args.id or f"run-{name.replace('_', '-')}"
    # Tie runs/ to the same knowledge base as nodes/ (honors KNOWLEDGE_DIR).
    run_dir = nodes_dir().parent / "runs" / node_id
    node_path = nodes_dir() / f"{node_id}.md"
    if (run_dir.exists() or node_path.exists()) and not args.force:
        p.error(f"{node_id} already registered (use --force)")
    run_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for fn in BUNDLE_FILES:
        src = repro / fn
        if src.exists():
            shutil.copy2(src, run_dir / fn)
            copied.append(fn)
    preds = repro / "predictions"
    for fn in PREDICTION_FILES:
        src = preds / fn
        if src.exists():
            shutil.copy2(src, run_dir / fn)
            copied.append(fn)

    try:
        rel_run_dir = run_dir.relative_to(repo_root())
    except ValueError:
        rel_run_dir = run_dir
    artifacts: dict[str, str] = {"run_dir": str(rel_run_dir)}
    if (run_dir / "pred_test.npz").exists():
        artifacts["predictions"] = str(rel_run_dir / "pred_test.npz")
    log = find_log(name)
    if log and log.exists():
        artifacts["log"] = str(log.relative_to(repo_root()))
    out_txt = repro / "output_dir.txt"
    if out_txt.exists():
        odir = out_txt.read_text(encoding="utf-8").strip()
        if odir:
            artifacts["output_dir"] = portable_output_dir(odir, run)

    repro_meta = {k: run[k] for k in ("commit", "branch", "remote") if run.get(k)}
    if cmd:
        repro_meta["command"] = cmd

    meta: dict = {"id": node_id, "type": "run", "title": name}
    if issue is not None:
        meta["issue"] = issue
    meta["provider"] = provider
    if run.get("session"):
        meta["session"] = run["session"]
    date = (run.get("captured_at") or "")[:10]
    if date:
        meta["date"] = date
    meta["status"] = "done"
    meta["config"] = config
    meta["metrics"] = metrics
    if repro_meta:
        meta["repro"] = repro_meta
    meta["artifacts"] = artifacts
    meta["parents"] = []
    meta["relations"] = []
    meta["tags"] = []

    body = (
        "## 考察 / Findings\n\n"
        f"<!-- run `{name}` の結果と考察を書く。parents/tags も埋め、"
        " 主要 metrics は frontmatter と一致させること。 -->\n"
    )
    node_path.parent.mkdir(parents=True, exist_ok=True)
    node_path.write_text(f"---\n{dump_frontmatter(meta)}---\n\n{body}", encoding="utf-8")

    def _rel(path: Path) -> Path:
        try:
            return path.relative_to(repo_root())
        except ValueError:
            return path

    print(f"registered {node_id}")
    print(f"  node:   {_rel(node_path)}")
    print(f"  bundle: {rel_run_dir} (copied: {', '.join(copied) or 'nothing'})")
    print(f"  metrics: {len(metrics)} | config: {list(config)}")
    print("Next: write the 考察 body, set parents/tags, then run kg_validate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
