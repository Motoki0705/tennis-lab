#!/usr/bin/env python
"""Check that every run's repro.sh references only scripts it can reproduce.

A script reference (``*.py``/``*.sh``/``*.yaml``/``*.yml`` file, ``python -m``
module, or ``PYTHONPATH`` entry) is reproducible when it is

- saved in a run bundle (``$SCRIPT_DIR/...`` or ``knowledge/runs/<run>/...``),
- tracked by git at the commit the repro.sh checks out (HEAD when it pins none),
- added by the bundle's ``uncommitted.patch``, or
- a checkout of this repository itself (a ``PYTHONPATH`` of the repo root).

Anything else (``/tmp``, ``outputs/``, an untracked build) is reported. Data
and checkpoint arguments are inputs, not scripts, and are not checked. A missing
reference is an ERROR (exit 1); a reference at a commit absent from this clone
is a WARN unless ``--strict``. ``KNOWLEDGE_DIR`` selects another knowledge root.

Usage:
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_repro_paths.py [--json] [--strict]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from kg_lib import repo_root

SCRIPT_SUFFIXES = (".py", ".sh", ".yaml", ".yml")
COMMAND_MARKER = "# --- original training command ---"
# Checkouts of this repository under other names (worktrees, earlier clones).
CHECKOUT_PATTERNS = (
    re.compile(r"^/home/[^/]+/projects/tennis-lab/\.claude/worktrees/[^/]+/?(?P<rel>.*)$"),
    re.compile(r"^/home/[^/]+/projects/tennis-lab-worktrees/[^/]+/?(?P<rel>.*)$"),
    re.compile(r"^/home/[^/]+/projects/tennis-lab/?(?P<rel>.*)$"),
)
# Shell spellings of the repository root inside a repro.sh (it cds there first).
ROOT_VARIABLES = ("$PWD", "${PWD}", "$REPO", "${REPO}", "$root", "${root}", "$task_repo_root", "${task_repo_root}")
# Spellings of the run bundle directory.
BUNDLE_VARIABLES = ("$SCRIPT_DIR", "${SCRIPT_DIR}", "$bundle", "${bundle}")
TEMPORARY_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)=\"?\$\(mktemp\b", re.MULTILINE)
CHECKOUT = re.compile(r"^(?:git checkout|git -C \S+ worktree add --detach \S+) ([0-9a-f]{7,40})\b", re.MULTILINE)


@dataclass(frozen=True)
class Finding:
    run: str
    reference: str
    kind: str
    status: str
    detail: str


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)


def _tracked(root: Path, commit: str, relative: str) -> bool | None:
    """Whether ``relative`` exists at ``commit``; ``None`` when the commit is unknown."""
    if _git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode:
        return None
    return _git(root, "cat-file", "-e", f"{commit}:{relative}").returncode == 0


def _patched(run_dir: Path) -> set[str]:
    patch = run_dir / "uncommitted.patch"
    if not patch.is_file():
        return set()
    return {line[6:].strip() for line in patch.read_text(errors="replace").splitlines() if line.startswith("+++ b/")}


def _command(text: str) -> str:
    return text.split(COMMAND_MARKER, 1)[1] if COMMAND_MARKER in text else text


def _references(command: str) -> list[tuple[str, str]]:
    """``(kind, value)`` script references of one shell command."""
    found: list[tuple[str, str]] = []
    for line in command.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError:
            tokens = line.split()
        for index, token in enumerate(tokens):
            if token.startswith("PYTHONPATH="):
                for entry in re.sub(r"\$\{PYTHONPATH:\+:?\$PYTHONPATH\}|\$\{?PYTHONPATH\}?", "", token.removeprefix("PYTHONPATH=")).split(":"):
                    if entry.rstrip("/") not in ("", ".", *ROOT_VARIABLES):
                        found.append(("pythonpath", entry))
                continue
            if token == "-m" and index + 1 < len(tokens):
                found.append(("module", tokens[index + 1]))
                continue
            value = token.split("=", 1)[1] if "=" in token and not token.startswith(("/", ".", "$")) else token
            interpreted = index > 0 and Path(tokens[index - 1]).name.startswith(("python", "bash"))
            if value.endswith(SCRIPT_SUFFIXES) and ("/" in value or interpreted):
                found.append(("file", value))
    return found


def _module_paths(module: str) -> tuple[str, ...]:
    base = module.replace(".", "/")
    return (f"{base}.py", f"{base}/__main__.py")


def check_run(root: Path, repro: Path) -> list[Finding]:
    text = repro.read_text()
    run_dir = repro.parent
    commit_match = CHECKOUT.search(text)
    commit = commit_match.group(1) if commit_match else "HEAD"
    patched = _patched(run_dir)
    temporary = {match.group("name") for match in TEMPORARY_ASSIGNMENT.finditer(text)}
    findings: list[Finding] = []
    for kind, value in _references(_command(text)):
        if any(value.startswith((f"${name}", f"${{{name}}}")) for name in temporary):
            status, detail = "ok", "generated by the repro.sh in a temporary directory"
        else:
            status, detail = _resolve(root, run_dir, commit, patched, kind, value)
        findings.append(Finding(run_dir.name, value, kind, status, detail))
    return findings


def _in_repository(root: Path, commit: str, patched: set[str], relative: str) -> tuple[str, str]:
    if relative in patched:
        return "ok", "added by the saved uncommitted.patch"
    if relative.startswith("knowledge/runs/"):
        return ("ok", "saved in a run bundle") if (root / relative).exists() else ("missing", "not in the named run bundle")
    tracked = _tracked(root, commit, relative)
    if tracked is None:
        return "unverifiable", f"target commit {commit} is not in this repository"
    return ("ok", f"tracked at {commit[:8]}") if tracked else ("missing", f"not tracked at {commit[:8]} (untracked or generated file)")


def _resolve(root: Path, run_dir: Path, commit: str, patched: set[str], kind: str, value: str) -> tuple[str, str]:
    if kind == "module":
        results = [_in_repository(root, commit, patched, relative) for relative in _module_paths(value)]
        return next((r for r in results if r[0] == "ok"), results[0])
    for variable in ROOT_VARIABLES:
        if value.startswith(f"{variable}/"):
            value = value.removeprefix(f"{variable}/")
    for variable in BUNDLE_VARIABLES:
        if value.startswith(f"{variable}/"):
            path = (run_dir / value.removeprefix(f"{variable}/")).resolve()
            if not path.is_relative_to(run_dir.parent.resolve()):
                return "missing", "points outside the knowledge run bundles"
            return ("ok", "saved in a run bundle") if path.is_file() else ("missing", f"not in the run bundle: {path.name}")
    relative = value
    for pattern in CHECKOUT_PATTERNS:
        match = pattern.match(value)
        if match:
            relative = match.group("rel")
            break
    if Path(relative).is_absolute():
        return "missing", "outside the repository and the run bundles"
    if kind == "pythonpath" and relative.rstrip("/") in ("", "."):
        return "ok", "repository root"
    return _in_repository(root, commit, patched, relative.rstrip("/"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print every finding as JSON")
    parser.add_argument("--strict", action="store_true", help="Also fail on references at commits absent from this clone")
    args = parser.parse_args()
    root = repo_root()
    runs = Path(os.environ.get("KNOWLEDGE_DIR") or root / "knowledge") / "runs"
    repros = sorted(runs.glob("*/repro.sh"))
    findings = [f for repro in repros for f in check_run(root, repro)]
    if args.json:
        print(json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=2))
    missing = [f for f in findings if f.status == "missing"]
    unverifiable = [f for f in findings if f.status == "unverifiable"]
    for f in missing:
        print(f"ERROR: {f.run}: {f.kind} {f.reference} ({f.detail})")
    for f in unverifiable:
        print(f"WARN: {f.run}: {f.kind} {f.reference} ({f.detail})")
    print(f"\n{len(repros)} repro.sh, {len(findings)} script references — {len(missing)} missing, {len(unverifiable)} unverifiable.")
    return 1 if missing or (args.strict and unverifiable) else 0


if __name__ == "__main__":
    raise SystemExit(main())
