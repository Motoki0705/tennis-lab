#!/usr/bin/env python3
"""Resolve the active checkout and connect ARIS to Tennis Lab's shared queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workspace:
    checkout: Path
    repository: Path

    @classmethod
    def discover(cls) -> Workspace:
        def git_path(*args: str) -> Path:
            result = subprocess.run(
                ["git", "rev-parse", *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return Path(result.stdout.strip()).resolve()

        checkout = git_path("--show-toplevel")
        common = git_path("--path-format=absolute", "--git-common-dir")
        return cls(checkout=checkout, repository=common.parent)

    @property
    def bundle(self) -> Path:
        return self.checkout / ".agents" / "aris"

    @property
    def queue(self) -> Path:
        return self.repository / ".training_queue"

    @property
    def manifest(self) -> Path:
        return self.checkout / ".aris" / "installed-skills-codex.txt"

    def inventory(self) -> dict[str, Any]:
        with (self.bundle / "upstream.toml").open("rb") as stream:
            return tomllib.load(stream)

    def paths(self) -> dict[str, str]:
        return {
            "checkout": str(self.checkout),
            "repository": str(self.repository),
            "aris_repo": str(self.bundle.resolve()),
            "queue": str(self.queue),
            "manifest": str(self.manifest),
        }

    def check_files(self) -> list[str]:
        inventory = self.inventory()
        required = [
            self.bundle / "INTEGRATION.md",
            self.bundle / "LICENSE",
            self.bundle / "tools" / "run_state.py",
            self.bundle / "tools" / "provenance.py",
            self.bundle / "tools" / "iteration_log.py",
            self.bundle / "templates" / "RESEARCH_BRIEF_TEMPLATE.md",
            self.checkout / ".agents/skills/shared-references/reviewer-routing.md",
            self.checkout / ".agents/skills/training-queue/scripts/training_queue.sh",
            self.checkout / ".agents/skills/knowledge-control/SKILL.md",
            self.checkout / ".agents/skills/render-html/scripts/render_html.py",
        ]
        required.extend(
            self.checkout / ".agents" / "skills" / name / "SKILL.md"
            for name in inventory["skills"]
        )
        required.extend(self.bundle / path for path in inventory["support_files"])
        return [
            f"Missing installed resource: {path}"
            for path in required
            if not path.is_file()
        ]

    def setup(self) -> None:
        errors = self.check_files()
        if errors:
            raise ValueError("\n".join(errors))
        inventory = self.inventory()
        # Only a runtime pointer for upstream helper resolution. Source selection
        # remains in upstream.toml, and setup never downloads or updates skills.
        content = (
            "version\t1\n"
            "installation\ttennis-lab-copy\n"
            f"repo_root\t{self.bundle.resolve()}\n"
            f"project_root\t{self.checkout}\n"
            f"revision\t{inventory['revision']}\n"
        )
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(content, encoding="utf-8")

    def doctor(self) -> dict[str, Any]:
        errors = self.check_files()
        if not self.manifest.is_file():
            errors.append(
                "Runtime manifest is missing; run .agents/aris/aris.py setup."
            )
        else:
            values = dict(
                line.split("\t", 1)
                for line in self.manifest.read_text(encoding="utf-8").splitlines()
                if "\t" in line
            )
            if values.get("repo_root") != str(self.bundle.resolve()):
                errors.append("Runtime manifest points to another checkout; run setup.")
            if values.get("project_root") != str(self.checkout):
                errors.append("Runtime manifest project root is stale; run setup.")
            if values.get("revision") != self.inventory()["revision"]:
                errors.append("Runtime manifest revision is stale; run setup.")
        return {
            "ok": not errors,
            "paths": self.paths(),
            "upstream_revision": self.inventory()["revision"],
            "skill_count": len(self.inventory()["skills"]),
            "errors": errors,
        }

    def run_queue(self, args: list[str]) -> int:
        if args and args[0] == "--":
            args = args[1:]
        if not args:
            raise ValueError("queue requires a training-queue subcommand")
        configured = os.environ.get("TRAINING_QUEUE_DIR")
        if configured and Path(configured).resolve() != self.queue:
            raise ValueError(
                f"TRAINING_QUEUE_DIR must use the shared repository queue: {self.queue}"
            )
        script = (
            self.checkout / ".agents/skills/training-queue/scripts/training_queue.sh"
        )
        if not script.is_file():
            raise ValueError(f"Training queue is unavailable: {script}")
        env = dict(os.environ)
        env["TRAINING_QUEUE_DIR"] = str(self.queue)
        # Let the existing worker own jobs, processes, cancellation and resources.
        # In particular, preserve the WORKTREE cwd in each job's repro bundle.
        return subprocess.run(
            ["bash", str(script), *args],
            cwd=self.checkout,
            env=env,
            check=False,
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("setup", help="Create checkout-local ARIS helper pointers")
    subparsers.add_parser(
        "doctor", help="Check installed resources and helper pointers"
    )
    subparsers.add_parser("paths", help="Print checkout, shared queue and helper paths")
    queue = subparsers.add_parser(
        "queue", help="Delegate to the existing training queue"
    )
    queue.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        workspace = Workspace.discover()
        if args.command == "queue":
            return workspace.run_queue(args.args)
        if args.command == "setup":
            workspace.setup()
        if args.command == "paths":
            print(json.dumps(workspace.paths(), indent=2, ensure_ascii=False))
            return 0
        report = workspace.doctor()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"ARIS: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
