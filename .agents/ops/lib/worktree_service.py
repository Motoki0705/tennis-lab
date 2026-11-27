"""Worktree setup and cleanup for transient ops services."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# The ops tree includes the runner, configuration, prompts and schemas. The
# worker follow-up's read-only queue reader is the only source import outside it.
OPS_SOURCE_PATHS = (
    ".agents/ops",
    "src/__init__.py",
    "src/utils/__init__.py",
    "src/utils/training_queue.py",
)


def _exec_property(setting: str, *argv: str | Path) -> str:
    # systemd-run parses transient Exec* properties as argv. The ":" prefix
    # disables environment substitution and specifiers are already literal.
    quoted = " ".join(json.dumps(str(arg), ensure_ascii=False) for arg in argv)
    return f"--property={setting}=:{quoted}"


def worktree_properties(
    checkout: Path, *, common: Path, head: str, venv: Path
) -> list[str]:
    checkout.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if checkout.exists() or checkout.is_symlink():
        raise ValueError(f"service checkout already exists: {checkout}")
    git, ln = shutil.which("git"), shutil.which("ln")
    if git is None or ln is None:
        raise ValueError("pinned checkout service requires git and ln")
    return [
        _exec_property(
            "ExecStartPre",
            git,
            f"--git-dir={common}",
            "worktree",
            "add",
            "--detach",
            "--no-checkout",
            checkout,
            head,
        ),
        _exec_property(
            "ExecStartPre",
            git,
            "-C",
            checkout,
            "restore",
            "--source=HEAD",
            "--worktree",
            "--",
            *OPS_SOURCE_PATHS,
        ),
        _exec_property(
            "ExecStartPre", ln, "--symbolic", "--", venv, checkout / ".venv"
        ),
        _exec_property(
            "ExecStopPost",
            git,
            f"--git-dir={common}",
            "worktree",
            "remove",
            "--force",
            checkout,
        ),
    ]
