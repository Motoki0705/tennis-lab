#!/usr/bin/env python
"""Register a finished Colab job from its Drive operation record.

The tennis-colab worker publishes each job to the Drive-root-relative record
``operations/colab/<session>/<job-id>/`` (``request.json``,
``source_changes.json``, ``status.json``, ``stdout.log``). Download that
directory with the tennis-drive skill and pass the local copy; this script never
reads Drive itself. It builds ``knowledge/runs/<id>/`` from the record:

- ``colab-request.json`` / ``colab-status.json``: the published records, unchanged;
- ``colab-source.json``: the source snapshot's commit and exclusion lists;
- ``uncommitted.patch``: the tracked diff plus untracked text files as additions;
- ``repro.sh``: checkout, patch and the exact argv from the recorded cwd;
- ``stdout.log`` (unless ``--skip-log``) and ``metrics.json`` (with ``--metrics``).

Outputs stay on Drive and are referenced, never copied. Every file is checked
against the run storage policy before anything is written.

``--attach`` adds the bundle to an existing run node instead (for runs recorded
before this route existed): it sets ``repro`` and the ``run_dir`` / ``drive_*``
artifacts, refuses to overwrite different values or bundle files, and keeps the
node's findings, status and metrics.

Usage:
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_from_colab.py RECORD_DIR \
        --drive-record operations/colab/<session>/<job-id> --task court_detection \
        --id run-i1343-scene-4-s42 --title "..." --provider codex
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from kg_artifacts import RUN_LIMIT, removal_reason
from kg_lib import (
    Node,
    config_from_command,
    load_nodes,
    nodes_dir,
    repo_root,
    test_metrics_from_log,
)
from kg_schema import PROVIDERS, iso_date, json_value, nonempty_text
from kg_storage import check_identity, prepare_node, registration_lock, write_node

TERMINAL = {"completed", "failed", "cancelled", "abandoned", "save_failed"}
DRIVE_RECORD = re.compile(r"^operations/colab/(?P<session>[^/\s]+)/(?P<job>[^/\s]+)$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_VM_ROOT = "/content/tennis-lab"


class RecordError(ValueError):
    """The Colab record is incomplete or inconsistent; nothing was written."""


def _json(path: Path, *, versioned: bool) -> dict[str, Any]:
    """Read a record file; request/status carry ``schema_version: 1``, the source snapshot none."""
    if not path.is_file():
        raise RecordError(f"{path.name} is missing from the record")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RecordError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or (versioned and data.get("schema_version") != 1):
        raise RecordError(f"{path.name} must be a JSON object{' of schema_version 1' if versioned else ''}")
    return data


def load_record(record: Path, drive_record: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    match = DRIVE_RECORD.fullmatch(drive_record)
    if not match:
        raise RecordError("--drive-record must be operations/colab/<session>/<job-id>")
    request = _json(record / "request.json", versioned=True)
    status = _json(record / "status.json", versioned=True)
    source = _json(record / "source_changes.json", versioned=False)
    job = request.get("job_id")
    if job != match.group("job") or status.get("job_id") != job:
        raise RecordError(f"job IDs differ between --drive-record, request.json and status.json ({match.group('job')}, {job}, {status.get('job_id')})")
    argv = request.get("argv")
    if not isinstance(argv, list) or not argv or not all(nonempty_text(a) for a in argv):
        raise RecordError("request.json argv must be a nonempty list of strings")
    if not COMMIT.fullmatch(str(request.get("source_commit"))) or source.get("commit") != request["source_commit"]:
        raise RecordError("request.json source_commit and source_changes.json commit must be the same full SHA")
    if not nonempty_text(request.get("cwd")) or not nonempty_text(request.get("created_at")):
        raise RecordError("request.json needs cwd and created_at")
    if status.get("status") not in TERMINAL:
        raise RecordError(f"job status is {status.get('status')!r}; register only after a terminal status")
    if status["status"] == "completed" and not status.get("drive_saved_at"):
        raise RecordError("a completed job must record drive_saved_at (outputs were persisted)")
    if not isinstance(source.get("patch"), str) or not isinstance(source.get("untracked"), dict):
        raise RecordError("source_changes.json needs a base64 patch and an untracked mapping")
    return request, status, source


def _addition(name: str, text: str) -> str:
    """A ``git apply``-able hunk creating ``name`` with ``text``."""
    lines = text.splitlines(keepends=True)
    header = f"diff --git a/{name} b/{name}\nnew file mode 100644\n"
    if not lines:
        return header
    body = "".join("+" + line if line.endswith("\n") else "+" + line + "\n\\ No newline at end of file\n" for line in lines)
    return header + f"--- /dev/null\n+++ b/{name}\n@@ -0,0 +1,{len(lines)} @@\n" + body


def source_patch(source: dict[str, Any]) -> str:
    try:
        tracked = base64.b64decode(source["patch"], validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise RecordError(f"source_changes.json patch is not base64 UTF-8 text: {exc}") from exc
    parts = [tracked] if tracked else []
    for name in sorted(source["untracked"]):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise RecordError(f"untracked path {name} leaves the repository")
        try:
            text = base64.b64decode(source["untracked"][name], validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RecordError(f"untracked file {name} is not UTF-8 text; commit it or reproduce it inside the command") from exc
        parts.append(_addition(name, text))
    return "".join(part if part.endswith("\n") else part + "\n" for part in parts)


def relative_cwd(cwd: str, vm_root: str) -> str:
    path, root = PurePosixPath(cwd), PurePosixPath(vm_root)
    if path != root and root not in path.parents:
        raise RecordError(f"recorded cwd {cwd} is outside the VM checkout {vm_root}; pass --vm-repo-root")
    return path.relative_to(root).as_posix()


def repro_script(request: dict[str, Any], drive_record: str, cwd: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by kg_from_colab.py from the Colab record {drive_record}.",
        "# Inputs and outputs live on Drive; fetch them with the tennis-drive skill as the task README describes.",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'REPO="${TENNIS_REPO:?set TENNIS_REPO to a tennis-lab checkout}"',
        'cd "$REPO"',
        f"git checkout {request['source_commit']}",
        'if [ -s "$SCRIPT_DIR/uncommitted.patch" ]; then git apply "$SCRIPT_DIR/uncommitted.patch"; fi',
    ]
    if cwd != ".":
        lines.append(f"cd {shlex.quote(cwd)}")
    lines += ["# --- original training command ---", shlex.join(request["argv"])]
    return "\n".join(lines) + "\n"


def load_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data or not json_value(data) or any(type(v) not in (int, float) or isinstance(v, bool) for v in data.values()):
        raise RecordError(f"{path} must be a nonempty JSON object of finite numbers")
    return {key: round(float(value), 6) for key, value in data.items()}


def bundle_location(run_dir: Path) -> str:
    try:
        return run_dir.relative_to(repo_root()).as_posix()
    except ValueError:
        return str(run_dir)


def attach(args: argparse.Namespace, request: dict[str, Any], files: dict[str, bytes], cwd: str) -> None:
    """Add the bundle to an existing run node; never overwrite different evidence."""
    matches: list[Node] = [n for n in load_nodes() if n.id == args.id]
    if len(matches) != 1 or matches[0].type != "run" or matches[0].meta.get("task") != args.task:
        raise RecordError(f"--attach needs exactly one existing run node {args.id} in task {args.task}")
    node = matches[0]
    run_dir = nodes_dir().parent / "runs" / args.id
    clash = sorted(name for name in files if (run_dir / name).exists())
    if clash:
        raise RecordError(f"the bundle already has {', '.join(clash)}; remove them deliberately or register a new node")
    existing = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()) if run_dir.exists() else 0
    if existing + sum(len(data) for data in files.values()) > RUN_LIMIT:
        raise RecordError("bundle rejected before writing: run exceeds 2 MiB")
    meta = dict(node.meta)
    outputs = [*request.get("persist", []), *request.get("runner_outputs", [])]
    proposed = {
        "repro": {"commit": request["source_commit"], "command": shlex.join(request["argv"]), "cwd": cwd},
        "artifacts": {"run_dir": bundle_location(run_dir), "drive_record": args.drive_record, **({"drive_outputs": outputs} if outputs else {})},
    }
    for field, values in proposed.items():
        current = meta.get(field) or {}
        if not isinstance(current, dict):
            raise RecordError(f"{node.path.name}: {field} must be a mapping")
        conflicts = sorted(k for k, v in values.items() if k in current and current[k] != v)
        if conflicts:
            raise RecordError(f"{node.path.name}: {field} already records different {', '.join(conflicts)}; check that the record belongs to this run")
        meta[field] = {**current, **values}
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (run_dir / name).write_bytes(data)
    (run_dir / "repro.sh").chmod(0o755)
    write_node(node.path, meta, node.body)
    print(f"attached {args.drive_record} to {args.id}")
    print(f"  bundle: {bundle_location(run_dir)} ({', '.join(sorted(files))})")
    print("  findings, status and metrics are unchanged; reconcile them with the record if needed")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("record", type=Path, help="local copy of the Drive record directory")
    p.add_argument("--drive-record", required=True, help="Drive-root-relative operations/colab/<session>/<job-id>")
    p.add_argument("--task", required=True)
    p.add_argument("--id", required=True, help="node id, e.g. run-i1343-scene-4-s42")
    p.add_argument("--title", help="node title (required unless --attach)")
    p.add_argument("--provider", choices=sorted(PROVIDERS))
    p.add_argument("--issue", type=int)
    p.add_argument("--date", help="experiment date; defaults to the job's created_at date")
    p.add_argument("--papers", nargs="*", default=[])
    p.add_argument("--metrics", type=Path, help="metric summary JSON (finite numbers) fetched from the run outputs")
    p.add_argument("--skip-log", action="store_true", help="do not copy stdout.log (it stays in the Drive record)")
    p.add_argument("--vm-repo-root", default=DEFAULT_VM_ROOT, help=f"checkout path on the VM (default: {DEFAULT_VM_ROOT})")
    p.add_argument("--force", action="store_true", help="replace an existing node and bundle files")
    p.add_argument("--attach", action="store_true", help="add the bundle to an existing run node, keeping its findings")
    args = p.parse_args()
    if args.issue is not None and args.issue <= 0:
        p.error("issue must be positive")
    node_fields = {"--title": args.title, "--provider": args.provider, "--issue": args.issue, "--date": args.date, "--papers": args.papers or None, "--force": args.force or None}
    if args.attach and any(value is not None for value in node_fields.values()):
        p.error(f"--attach keeps the node's metadata; drop {', '.join(k for k, v in node_fields.items() if v is not None)}")
    if not args.attach and not args.title:
        p.error("--title is required for a new node")
    try:
        request, status, source = load_record(args.record, args.drive_record)
        date = args.date or str(request["created_at"])[:10]
        if not iso_date(date):
            raise RecordError("experiment date must be YYYY-MM-DD; pass --date")
        cwd = relative_cwd(request["cwd"], args.vm_repo_root)
        log = args.record / "stdout.log"
        log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else None
        if log_text is None and not args.skip_log:
            raise RecordError("stdout.log is missing; pass --skip-log to register without it")
        metrics = load_metrics(args.metrics) if args.metrics else test_metrics_from_log(log_text or "")
        files: dict[str, bytes] = {
            "colab-request.json": (args.record / "request.json").read_bytes(),
            "colab-status.json": (args.record / "status.json").read_bytes(),
            "colab-source.json": (json.dumps({k: source.get(k) for k in ("commit", "excluded", "storage_excluded", "exclusion_reasons")} | {"untracked": sorted(source["untracked"])}, ensure_ascii=False, indent=2) + "\n").encode(),
            "repro.sh": repro_script(request, args.drive_record, cwd).encode(),
        }
        patch = source_patch(source)
        if patch:
            files["uncommitted.patch"] = patch.encode()
        if not args.skip_log and log_text is not None:
            files["stdout.log"] = log.read_bytes()
        if args.metrics:
            files["metrics.json"] = (json.dumps(metrics, indent=2) + "\n").encode()
        for name, data in files.items():
            reason = removal_reason(Path(name), len(data))
            if reason:
                hint = "; pass --skip-log (the log stays in the Drive record)" if name == "stdout.log" else ""
                raise RecordError(f"bundle rejected before writing: {name}: {reason}{hint}")
    except (RecordError, OSError, ValueError) as exc:
        p.error(str(exc))

    if args.attach:
        with registration_lock():
            try:
                attach(args, request, files, cwd)
            except ValueError as exc:
                p.error(str(exc))
        return 0
    meta: dict[str, Any] = {"id": args.id, "type": "run", "task": args.task, "papers": args.papers}
    with registration_lock():
        try:
            check_identity(meta)
            run_dir = nodes_dir().parent / "runs" / args.id
            existing = {path.relative_to(run_dir).as_posix(): path.stat().st_size for path in run_dir.rglob("*") if path.is_file()}
            if sum({**existing, **{k: len(v) for k, v in files.items()}}.values()) > RUN_LIMIT:
                raise RecordError("bundle rejected before writing: run exceeds 2 MiB")
            node_path = prepare_node(meta, args.force)
        except ValueError as exc:
            p.error(str(exc))
        if (run_dir.exists() or node_path.exists()) and not args.force:
            p.error(f"{args.id} already registered (use --force)")
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (run_dir / name).write_bytes(data)
        (run_dir / "repro.sh").chmod(0o755)
        rel_run_dir = bundle_location(run_dir)
        outputs = [*request.get("persist", []), *request.get("runner_outputs", [])]
        artifacts: dict[str, Any] = {"run_dir": rel_run_dir, "drive_record": args.drive_record}
        if outputs:
            artifacts["drive_outputs"] = outputs
        meta.update(title=args.title, date=date)
        if args.issue is not None:
            meta["issue"] = args.issue
        if args.provider:
            meta["provider"] = args.provider
        meta["status"] = "done" if status["status"] == "completed" and status.get("returncode") == 0 else "failed"
        command = shlex.join(request["argv"])
        meta.update(config=config_from_command(command), metrics=metrics,
                    repro={"commit": request["source_commit"], "command": command, "cwd": cwd},
                    artifacts=artifacts, parents=[], relations=[], tags=[])
        body = (
            "## 考察 / Findings\n\n"
            f"<!-- Colab job `{request['job_id']}`（{status['status']}、returncode {status.get('returncode')}）の結果と考察を書く。"
            " 主要metricsはfrontmatterと一致させる。 -->\n"
        )
        write_node(node_path, meta, body)
    print(f"registered {args.id} from {args.drive_record}")
    print(f"  node:    {node_path}")
    print(f"  bundle:  {rel_run_dir} ({', '.join(sorted(files))})")
    print(f"  status:  {meta['status']} (Colab {status['status']}, returncode {status.get('returncode')})")
    print(f"  metrics: {len(metrics)} from {'--metrics' if args.metrics else 'stdout.log test tables'}")
    if outputs:
        print(f"  outputs stay on Drive: {', '.join(outputs)}")
    print("Next: write the findings, link parents/relations, declare summary_impact, then validate with --base-ref")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
