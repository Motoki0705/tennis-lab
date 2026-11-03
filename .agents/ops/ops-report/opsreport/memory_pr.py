"""Publication journal and a narrow, non-blocking merge path for report memory.

The caller holds schedule.lock (including triage). Only a published issue matching
the prepared report and author authorizes git/PR writes. CI is inspected once per
invocation; no agent or PR-review gate is invoked here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from . import memory, schedule, shell

REVIEW_DIR = Path(__file__).resolve().parents[2] / "review"
sys.path.insert(0, str(REVIEW_DIR))
from review_policy import CheckState, normalize_rollup  # noqa: E402

LOG = logging.getLogger("ops_report.memory")
REQUIRED_CHECKS_FILE = Path(__file__).resolve().parents[1] / "required_checks.txt"
PR_FIELDS = (
    "number,url,state,body,headRefName,headRefOid,baseRefName,baseRefOid,"
    "isDraft,isCrossRepository,mergeable,statusCheckRollup"
)
TERMINAL = {"merged", "unchanged", "failed"}


class ObsoletePR(ValueError):
    """This proposal cannot be applied; the next report must use current main."""


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as fh:
        tmp = Path(fh.name)
        try:
            json.dump(value, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


class Store:
    def __init__(self, repo_root: Path, path: Path | None = None) -> None:
        self.repo_root = repo_root
        self.path = path or schedule.state_dir() / "memory.json"
        self.records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if (
                    set(data) != {"version", "records"}
                    or type(data["version"]) is not int
                    or data["version"] != 2
                ):
                    raise ValueError("invalid journal schema/version")
                if not isinstance(data["records"], dict):
                    raise ValueError("records must be an object")
                self.records = data["records"]
                for slot, row in self.records.items():
                    if (
                        not re.fullmatch(r"\d{8}", slot)
                        or date.fromisoformat(slot).strftime("%Y%m%d") != slot
                    ):
                        raise ValueError("invalid journal slot")
                    if row["state"] not in {"prepared", "pending", *TERMINAL}:
                        raise ValueError("invalid journal state")
                    publication = row["publication"]
                    if (
                        set(publication) != {"title", "body_sha256", "author"}
                        or any(
                            not isinstance(value, str) or not value
                            for value in publication.values()
                        )
                        or not re.fullmatch(r"[a-f0-9]{64}", publication["body_sha256"])
                    ):
                        raise ValueError("invalid journal publication identity")
                    memory.parse_update(row["update"])
                    if row["update"] is not None:
                        if not re.fullmatch(r"[a-f0-9]{40}", row["base_sha"]):
                            raise ValueError("invalid journal base SHA")
                        expected = memory.read_tree(repo_root, row["base_sha"])
                        if row["files"] != expected:
                            raise ValueError(
                                "journal memory differs from its main version"
                            )
                        memory.apply_update(
                            expected,
                            row["update"],
                            source="validate",
                            today=date.today(),
                        )
                    if row["head"] and not re.fullmatch(r"[a-f0-9]{40}", row["head"]):
                        raise ValueError("invalid journal head SHA")
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"corrupt memory journal at {self.path}: {exc}; repair explicitly"
                ) from exc

    def save(self) -> None:
        _atomic_json(self.path, {"version": 2, "records": self.records})

    def prepare(
        self,
        slot: str,
        dedicated: dict[str, Any],
        update: dict[str, Any] | None,
        *,
        title: str,
        body: str,
    ) -> None:
        old = self.records.get(slot)
        if old and old["state"] != "prepared":
            raise ValueError(
                f"memory for {slot} already published; never replace its observations"
            )
        # Validate before publishing. A malformed patch must not create an issue.
        memory.apply_update(
            dedicated["files"], update, source=f"ops-report:{slot}", today=date.today()
        )
        author = shell.gh_json(["api", "user"], cwd=self.repo_root)["login"]
        if not isinstance(author, str) or not author:
            raise ValueError("publication author must be an authenticated GitHub login")
        self.records[slot] = {
            "state": "prepared",
            "stage": "publication_unconfirmed",
            "error": None,
            "base_sha": dedicated.get("base_sha"),
            "files": dedicated["files"],
            "update": update,
            "publication": {
                "title": title,
                "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "author": author,
            },
            "report_url": None,
            "head": None,
            "pr_url": None,
        }
        self.save()

    def record_publication(self, slot: str, url: str) -> None:
        """Bind a create response or edit target without confirming publication."""
        if not re.fullmatch(
            r"https://github\.com/[^/\s]+/[^/\s]+/issues/[1-9]\d*", url
        ):
            raise ValueError("publication requires a valid report issue URL")
        row = self.records[slot]
        if row["state"] != "prepared":
            raise ValueError(f"memory {slot} publication is already confirmed")
        row["report_url"] = url
        self.save()

    def describe(self, history: dict[str, Any]) -> list[dict[str, Any]]:
        slots = {r["slot"]: r["url"] for r in history["reports"][-2:]}
        slots.update(
            {
                s: r["report_url"]
                for s, r in self.records.items()
                if r["state"] not in TERMINAL
            }
        )
        return [
            {
                "slot": slot,
                "report_url": url,
                **{
                    key: self.records.get(slot, {}).get(
                        key,
                        "記録なし（公開後の中断を要確認）" if key == "state" else None,
                    )
                    for key in ("state", "stage", "pr_url", "error")
                },
            }
            for slot, url in sorted(slots.items())
        ]

    def sync(self, history: dict[str, Any], run_dir: Path) -> list[str]:
        errors = []
        published = {r["slot"]: r for r in history["reports"]}
        for slot, row in sorted(self.records.items()):
            if row["state"] in TERMINAL:
                continue
            if slot not in published:
                if row["report_url"]:
                    raise ValueError(
                        f"memory {slot} has no published report; repair explicitly"
                    )
                continue  # failed publication: never write these observations
            published_report = published[slot]
            report_url = published_report["url"]
            if row["report_url"] and row["report_url"] != report_url:
                raise ValueError(f"memory {slot} publication URL mismatch")
            publication = row["publication"]
            if published_report["author"] != publication["author"]:
                raise ValueError(f"memory {slot} publication identity/content mismatch")
            if published_report["status"] != "complete":
                if row["state"] != "prepared":
                    raise ValueError(
                        f"memory {slot} has no published report; repair explicitly"
                    )
                # A failed edit may still expose the original incomplete body.
                continue
            if row["state"] == "prepared" and any(
                published_report[key] != publication[key]
                for key in ("title", "body_sha256")
            ):
                raise ValueError(f"memory {slot} publication identity/content mismatch")
            # Once verified, humans may edit the title and proposal checkboxes.
            row.update(state="pending", report_url=report_url, error=None)
            self.save()
            try:
                self._persist(slot, row, run_dir)
            except (shell.CommandError, ValueError, OSError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, ObsoletePR):
                    row["state"] = "failed"
                errors.append(f"memory {slot}: {row['error']}")
                LOG.error("%s", errors[-1])
            self.save()
        (run_dir / "memory_result.json").write_text(
            json.dumps(self.describe(history), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return errors

    def _persist(self, slot: str, row: dict[str, Any], run_dir: Path) -> None:
        if row["update"] is None:
            row.update(state="unchanged", stage="no_change")
            return
        branch = f"ops-report-memory/{slot}"
        expected = memory.apply_update(
            row["files"], row["update"], source=row["report_url"], today=date.today()
        )
        if expected == row["files"]:
            row.update(state="unchanged", stage="no_change")
            return
        if not row["head"]:
            row["stage"] = "commit"
            self.save()
            row["head"] = self._commit(slot, branch, row, expected)
            self.save()  # persist exact HEAD before any push
        self._verify_commit(row, expected)
        if not row["pr_url"]:
            row["stage"] = "push"
            self.save()
            shell.git(
                ["push", "origin", f"{row['head']}:refs/heads/{branch}"],
                cwd=self.repo_root,
            )
            row["stage"] = "pr_create"
            self.save()
            existing = shell.gh_list(
                ["pr", "list", "--state", "all", "--head", branch, "--json", "url"],
                limit=100,
                cwd=self.repo_root,
            )
            if len(existing) > 1:
                raise ValueError(f"multiple memory PRs for {slot}; resolve explicitly")
            if existing:
                row["pr_url"] = existing[0]["url"]
            else:
                body_file = run_dir / f"memory-pr-{slot}.md"
                body_file.write_text(
                    self._marker(slot, row) + f"\n\n{row['report_url']} の継続観察。\n"
                    f"変更は `{memory.DIRECTORY}/` のみ。必須CI通過後に専用処理がHEADを照合してマージする。\n",
                    encoding="utf-8",
                )
                pr_url = shell.gh(
                    [
                        "pr",
                        "create",
                        "--base",
                        "main",
                        "--head",
                        branch,
                        "--title",
                        f"運用レポート専用memory {slot}",
                        "--body-file",
                        str(body_file),
                    ],
                    cwd=self.repo_root,
                ).strip()
                if not re.fullmatch(
                    r"https://github\.com/[^/]+/[^/]+/pull/\d+", pr_url
                ):
                    raise ValueError("gh pr create printed no memory PR URL")
                row["pr_url"] = pr_url
            row["stage"] = "ci_wait"
            return  # first publication never waits/polls for CI
        self._merge(slot, branch, row)

    @staticmethod
    def _marker(slot: str, row: dict[str, Any]) -> str:
        return f"<!-- ops-report-memory:v1 slot={slot} base={row['base_sha']} head={row['head']} -->"

    def _common_dir(self, cwd: Path) -> Path:
        return Path(
            shell.git(
                ["rev-parse", "--path-format=absolute", "--git-common-dir"],
                cwd=cwd,
            ).strip()
        ).resolve()

    def _worktree(self, slot: str) -> Path:
        common = self._common_dir(self.repo_root)
        if common.name != ".git":
            raise ValueError(
                "memory publication requires a non-bare repository with .git"
            )
        parent = common.parent / ".claude/worktrees"
        if parent.resolve() != parent:
            raise ValueError("memory worktree parent must not be a symlink")
        return parent / f"ops-report-memory-{slot}"

    def _commit(
        self, slot: str, branch: str, row: dict[str, Any], expected: dict[str, str]
    ) -> str:
        worktree = self._worktree(slot)
        if not worktree.exists():
            # A branch left without its worktree is not overwritten silently.
            worktree.parent.mkdir(parents=True, exist_ok=True)
            shell.git(
                ["worktree", "add", "-b", branch, str(worktree), row["base_sha"]],
                cwd=self.repo_root,
            )
        if (
            worktree.is_symlink()
            or self._common_dir(worktree) != self._common_dir(self.repo_root)
            or shell.git(["branch", "--show-current"], cwd=worktree).strip() != branch
        ):
            raise ValueError(f"unexpected memory worktree: {worktree}")
        if shell.git(
            ["status", "--porcelain", "--untracked-files=all"], cwd=worktree
        ).strip():
            raise ValueError(
                f"interrupted/dirty memory worktree {worktree}; inspect and repair explicitly"
            )
        head: str = shell.git(["rev-parse", "HEAD"], cwd=worktree).strip()
        if head != row["base_sha"]:
            # Recover interruption between commit and journal update, but only if
            # that commit has exactly our parent and expected contents.
            self._verify_commit({**row, "head": head}, expected)
            return head
        target = worktree / memory.DIRECTORY
        if target.is_symlink():
            raise ValueError("dedicated memory directory must not be a symlink")
        for name in row["files"].keys() - expected.keys():
            (target / name).unlink()
        for name, text in expected.items():
            (target / name).write_text(text, encoding="utf-8")
        shell.git(["add", "--", memory.DIRECTORY], cwd=worktree)
        shell.run_process_group(
            ["git", "commit", "-m", f"Update ops report memory for {slot}"],
            cwd=worktree,
            timeout=600,
        )
        committed: str = shell.git(["rev-parse", "HEAD"], cwd=worktree).strip()
        return committed

    def _verify_commit(self, row: dict[str, Any], expected: dict[str, str]) -> None:
        head, base = row["head"], row["base_sha"]
        parents = shell.git(
            ["rev-list", "--parents", "-n", "1", head], cwd=self.repo_root
        ).split()
        if parents != [head, base]:
            raise ValueError("memory commit must have exactly its pinned main parent")
        paths = shell.git(
            ["diff", "--no-renames", "--name-only", "-z", base, head],
            cwd=self.repo_root,
        ).split("\0")
        if not any(paths) or any(
            p and not p.startswith(memory.DIRECTORY + "/") for p in paths
        ):
            raise ValueError(
                "memory PR changes must be confined to .agents/memory/ops-report/**"
            )
        if memory.read_tree(self.repo_root, head) != expected:
            raise ValueError("memory commit differs from the published observations")

    def _view(self, slot: str, branch: str, row: dict[str, Any]) -> dict[str, Any]:
        pr: dict[str, Any] = shell.gh_json(
            ["pr", "view", row["pr_url"], "--json", PR_FIELDS], cwd=self.repo_root
        )
        if (
            pr["url"] != row["pr_url"]
            or pr["headRefOid"] != row["head"]
            or pr["headRefName"] != branch
            or pr["baseRefName"] != "main"
            or pr["isCrossRepository"] is not False
            or pr["isDraft"] is not False
            or pr["body"].split("\n", 1)[0] != self._marker(slot, row)
        ):
            raise ValueError(
                "memory PR identity/HEAD/path contract changed; manual repair required"
            )
        return pr

    @staticmethod
    def _ci_ready(pr: dict[str, Any]) -> bool:
        if pr["mergeable"] not in {"MERGEABLE", "UNKNOWN", "CONFLICTING"}:
            raise ValueError(f"unknown memory PR mergeable state: {pr['mergeable']!r}")
        required = [
            line.split("#", 1)[0].strip()
            for line in REQUIRED_CHECKS_FILE.read_text().splitlines()
        ]
        required = [name for name in required if name]
        if not required:
            raise ValueError("memory merge requires a non-empty required_checks.txt")
        checks = normalize_rollup(pr["statusCheckRollup"] or ())
        # Required CI must be an actual successful check run. Keep the existing
        # latest-run normalization, including pessimism for unordered re-runs.
        required_runs = normalize_rollup(
            {**run, "conclusion": "FAILURE"}
            if run.get("conclusion") in {"NEUTRAL", "SKIPPED"}
            else run
            for run in pr["statusCheckRollup"] or ()
            if run.get("__typename") == "CheckRun" and run.get("name") in required
        )
        if any(
            check.state == CheckState.FAILURE for check in (*checks, *required_runs)
        ):
            raise ValueError("memory PR CI failed; main memory was not changed")
        return (
            pr["mergeable"] == "MERGEABLE"
            and set(required) <= {check.name for check in required_runs}
            and all(
                check.state == CheckState.SUCCESS for check in (*checks, *required_runs)
            )
        )

    def _merge(self, slot: str, branch: str, row: dict[str, Any]) -> None:
        from .collect import pin_base

        pr = self._view(slot, branch, row)
        main = pin_base(self.repo_root, "origin/main")
        if pr["state"] == "MERGED":
            shell.git(
                ["merge-base", "--is-ancestor", row["head"], main.sha],
                cwd=self.repo_root,
            )
            row.update(state="merged", stage="merged")
            self._remove_clean_worktree(slot, row["head"])
            return
        if pr["state"] != "OPEN":
            raise ObsoletePR(
                "memory PR closed without merging; main memory was not changed"
            )
        if memory.read_tree(self.repo_root, main.sha) != row["files"]:
            raise ObsoletePR(
                "main memory advanced after observation; regenerate from its latest version"
            )
        if pr["mergeable"] == "CONFLICTING":
            raise ValueError("memory PR conflicts; main memory was not changed")
        row["stage"] = "ci_wait"
        if not self._ci_ready(pr):
            return
        # Fresh HEAD and CI immediately before the atomic --match-head merge.
        fresh = self._view(slot, branch, row)
        if (
            fresh["state"] != "OPEN"
            or fresh["baseRefOid"] != main.sha
            or not self._ci_ready(fresh)
        ):
            return
        shell.gh(
            [
                "pr",
                "merge",
                row["pr_url"],
                "--merge",
                "--match-head-commit",
                row["head"],
            ],
            cwd=self.repo_root,
        )
        # Success is recorded only after GitHub and origin/main confirm it.
        merged = self._view(slot, branch, row)
        main = pin_base(self.repo_root, "origin/main")
        if merged["state"] != "MERGED":
            raise ValueError("memory merge not confirmed")
        shell.git(
            ["merge-base", "--is-ancestor", row["head"], main.sha], cwd=self.repo_root
        )
        row.update(state="merged", stage="merged")
        self._remove_clean_worktree(slot, row["head"])

    def _remove_clean_worktree(self, slot: str, head: str) -> None:
        path = self._worktree(slot)
        if (
            path.exists()
            and shell.git(["rev-parse", "HEAD"], cwd=path).strip() == head
            and not shell.git(
                ["status", "--porcelain", "--untracked-files=all"], cwd=path
            ).strip()
        ):
            shell.git(["worktree", "remove", str(path)], cwd=self.repo_root)
