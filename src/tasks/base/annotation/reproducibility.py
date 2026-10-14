"""The source checkout must still match the commit and diff frozen in the plan."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .plan import CampaignPlan


def require_frozen_code(root: Path, plan: CampaignPlan) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=root, check=True, capture_output=True
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if (
        commit != plan.code_commit
        or hashlib.sha256(diff).hexdigest() != plan.code_diff_digest
        or untracked
    ):
        raise ValueError(
            "source checkout no longer matches the frozen commit/diff, or has untracked files"
        )
