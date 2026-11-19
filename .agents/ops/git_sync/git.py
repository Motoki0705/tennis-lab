"""Git preflight and non-recursive, non-destructive fast-forward operations."""

from __future__ import annotations

import os
import stat
import subprocess
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


class Skip(RuntimeError):
    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


class Repository:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.ref = f"refs/tennis-lab/git-sync/{uuid.uuid4().hex}"

    def git(self, *args: str) -> str:
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "submodule.recurse=false",
                *args,
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            errors="surrogateescape",
            check=True,
            timeout=120,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        return result.stdout

    def validate_main(self) -> None:
        git_dir = self.git("rev-parse", "--path-format=absolute", "--git-dir").strip()
        common = self.git(
            "rev-parse", "--path-format=absolute", "--git-common-dir"
        ).strip()
        top = self.git("rev-parse", "--show-toplevel").strip()
        if git_dir != common or Path(top) != self.root:
            raise Skip("not_main_checkout", checkout=str(self.root))

    def preflight(self) -> str:
        self.validate_main()
        branch = self.git("rev-parse", "--abbrev-ref", "HEAD").strip()
        if branch != "main":
            raise Skip("non_main_branch", branch=branch)
        operations = []
        for name in (
            "MERGE_HEAD",
            "CHERRY_PICK_HEAD",
            "REVERT_HEAD",
            "rebase-merge",
            "rebase-apply",
            "sequencer",
            "BISECT_LOG",
            "BISECT_START",
            "index.lock",
            "HEAD.lock",
            "refs/heads/main.lock",
        ):
            path = Path(self.git("rev-parse", "--git-path", name).strip())
            if not path.is_absolute():
                path = self.root / path
            if path.exists():
                operations.append(name)
        if operations:
            raise Skip("git_operation_in_progress", operations=operations)
        flagged = [
            entry[2:]
            for entry in self.git("ls-files", "-v", "-z").split("\0")
            if entry and (entry[0].islower() or entry[0] == "S")
        ]
        if flagged:
            raise Skip("index_flags", paths=flagged)
        # Working gitlinks may deliberately lag a previous sync. Their checkout
        # is not ours to update; staged gitlink changes are still local edits.
        paths = set(
            self.git("diff", "--name-only", "-z", "--ignore-submodules=all").split("\0")
        )
        paths.update(
            self.git(
                "diff", "--cached", "--name-only", "-z", "--ignore-submodules=none"
            ).split("\0")
        )
        paths.discard("")
        if paths:
            raise Skip("tracked_changes", paths=sorted(paths))
        return self.git("rev-parse", "HEAD^{commit}").strip()

    def fetch(self) -> str:
        # FETCH_HEAD and origin/main are mutable by unrelated fetches. Only this
        # invocation owns this ref; keep the resulting object ID for every check.
        self.git(
            "fetch",
            "--no-recurse-submodules",
            "--no-tags",
            "origin",
            f"+refs/heads/main:{self.ref}",
        )
        return self.git("rev-parse", f"{self.ref}^{{commit}}").strip()

    def ancestor(self, older: str, newer: str) -> bool:
        try:
            self.git("merge-base", "--is-ancestor", older, newer)
            return True
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 1:
                raise
            return False

    def tree(self, revision: str) -> dict[str, tuple[str, str]]:
        result = {}
        for entry in self.git("ls-tree", "-rz", "--full-tree", revision).split("\0"):
            if entry:
                header, path = entry.split("\t", 1)
                mode, _, sha = header.split()
                result[path] = (mode, sha)
        return result

    def changes(self, head: str, target: str) -> dict[str, Any]:
        before, after = self.tree(head), self.tree(target)
        paths = sorted(
            p for p in before.keys() | after.keys() if before.get(p) != after.get(p)
        )
        dependencies, submodules = [], []
        for path in paths:
            name = PurePosixPath(path).name
            if name in {
                "uv.lock",
                "pyproject.toml",
                "poetry.lock",
                "Pipfile",
                "Pipfile.lock",
                ".python-version",
            } or (
                name.startswith(("requirements", "environment"))
                and name.endswith((".txt", ".in", ".yml", ".yaml"))
            ):
                dependencies.append(path)
            old, new = before.get(path), after.get(path)
            if (old and old[0] == "160000") or (new and new[0] == "160000"):
                submodules.append(
                    {
                        "path": path,
                        "before": old[1] if old else None,
                        "after": new[1] if new else None,
                    }
                )
        return {"dependency_changes": dependencies, "submodule_changes": submodules}

    def check_target(self, head: str, target: str) -> None:
        if not self.ancestor(head, target):
            raise Skip("local_ahead" if self.ancestor(target, head) else "diverged")
        before, after = self.tree(head), self.tree(target)
        incoming = {p: v for p, v in after.items() if before.get(p) != v}
        protected = [
            p
            for p in before.keys() | after.keys()
            if (
                p == ".venv"
                or p.startswith(".venv/")
                or p.startswith(".claude/worktrees/")
            )
            and before.get(p) != after.get(p)
        ]
        if protected:
            raise Skip("protected_path_change", paths=sorted(protected))
        layouts = [
            p
            for p in before.keys() | after.keys()
            if (
                (before.get(p, (None,))[0] == "160000")
                != (after.get(p, (None,))[0] == "160000")
            )
        ]
        # A gitlink pointer can change without touching its checkout. A type
        # conversion/addition/removal cannot promise that, so leave it to a human.
        if layouts:
            raise Skip("submodule_layout_change", paths=sorted(layouts))
        collisions = set()
        for path in incoming:
            parts = PurePosixPath(path).parts
            for length in range(1, len(parts) + 1):
                prefix = "/".join(parts[:length])
                candidate = self.root / prefix
                try:
                    mode = candidate.lstat().st_mode
                except FileNotFoundError:
                    break
                if prefix in before:
                    # A tracked symlink/file may become a directory; do not
                    # follow it into an external target during preflight.
                    break
                if not stat.S_ISDIR(mode) or length == len(parts):
                    tracked_directory = any(x.startswith(prefix + "/") for x in before)
                    # A large ignored data directory need not be enumerated if
                    # the entire untracked path would have to be replaced.
                    if not tracked_directory or self.git(
                        "ls-files", "--others", "--directory", "-z", "--", prefix
                    ):
                        collisions.add(prefix)
                    break
        if collisions:
            raise Skip("untracked_collision", paths=sorted(collisions))

    def merge(self, target: str) -> str:
        return self.git(
            "-c",
            "merge.autoStash=false",
            "merge",
            "--ff-only",
            "--no-autostash",
            "--no-edit",
            "--no-overwrite-ignore",
            target,
        )

    def release(self) -> None:
        self.git("update-ref", "-d", self.ref)
