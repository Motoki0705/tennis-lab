"""Read-only inspection for PM and parent sessions; writes only via parent CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from parent.github import GitHub
from parent.policy import issue_number

from .github import Projects
from .operations import inspection, observe
from .schema import OWNER, PROJECT_NUMBER


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect",))
    parser.add_argument("--repo")
    parser.add_argument("--parent", required=True)
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--project", type=int, default=PROJECT_NUMBER)
    args = parser.parse_args(argv)
    try:
        gh = GitHub(args.repo)
        result = inspection(
            observe(
                gh,
                Projects(),
                issue_number(args.parent, gh.repo),
                args.owner,
                args.project,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print(f"project-progress: {exc}", file=sys.stderr)
        return 2


raise SystemExit(main())
