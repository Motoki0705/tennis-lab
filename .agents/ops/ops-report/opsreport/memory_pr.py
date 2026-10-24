"""Publish report memory and hand it to the ordinary issue/review/merge loop.

The caller holds schedule.lock. No CI polling, reviewer launch or merge occurs
here. The ordinary gate revalidates this journal immediately before merging.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parent.execution import requirement_body  # noqa: E402

LOG = logging.getLogger("ops_report.memory")
PR_FIELDS = (
    "number,url,state,body,headRefName,headRefOid,baseRefName,baseRefOid,"
    "isDraft,isCrossRepository,mergeable"
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
                    or data["version"] not in {2, 3}
                ):
                    raise ValueError("invalid journal schema/version")
                if not isinstance(data["records"], dict):
                    raise ValueError("records must be an object")
                self.records = data["records"]
                if data["version"] == 2:
                    # Keep pinned observations/HEAD/terminal outcomes. The former
                    # waiting stage becomes a handoff, never a legacy merge path.
                    for row in self.records.values():
                        row["work_issue"] = None
                        row["work_issue_create_attempted"] = False
                        row["validation"] = None
                        if row["stage"] == "ci_wait":
                            row["stage"] = "handoff_pending"
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
                    if row["work_issue"] is not None and not re.fullmatch(
                        r"https://github\.com/[^/\s]+/[^/\s]+/issues/[1-9]\d*",
                        row["work_issue"],
                    ):
                        raise ValueError("invalid work issue URL")
                    if type(row["work_issue_create_attempted"]) is not bool:
                        raise ValueError("invalid work issue creation journal")
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
        _atomic_json(self.path, {"version": 3, "records": self.records})

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
            "work_issue": None,
            "work_issue_create_attempted": False,
            "validation": None,
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
                    for key in ("state", "stage", "pr_url", "work_issue", "error")
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
        row["validation"] = self._validation(row, expected)
        self.save()
        # Closed/merged legacy PRs are reconciled before creating a new work item.
        if row["pr_url"] and self._reconcile(slot, branch, row):
            return
        self._ensure_work_issue(slot, row, run_dir)
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
                    self._pr_body(slot, row),
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
        pr = self._view(slot, branch, row)
        self._link_issue(slot, row, pr, run_dir)
        row["stage"] = "review_handoff"
        self.save()

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
            ["git", "commit", "-m", f"Update ops report memory for {slot} [skip ci]"],
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
        shell.git(["diff", "--check", base, head], cwd=self.repo_root)
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

    def _reconcile(self, slot: str, branch: str, row: dict[str, Any]) -> bool:
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
            return True
        self._check_open(row, pr, main.sha)
        return False

    def _check_open(
        self, row: dict[str, Any], pr: dict[str, Any], main_sha: str
    ) -> None:
        if pr["state"] != "OPEN":
            raise ObsoletePR(
                "memory PR closed without merging; main memory was not changed"
            )
        if memory.read_tree(self.repo_root, main_sha) != row["files"]:
            raise ObsoletePR(
                "main memory advanced after observation; regenerate from its latest version"
            )
        if pr["mergeable"] == "CONFLICTING":
            raise ValueError("memory PR conflicts; main memory was not changed")

    @staticmethod
    def _validation(row: dict[str, Any], expected: dict[str, str]) -> dict[str, str]:
        return {
            "head": row["head"],
            "files_sha256": hashlib.sha256(
                json.dumps(expected, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            "checks": "memory.apply_update; exact parent/path/tree; git diff --check",
        }

    @staticmethod
    def _work_marker(slot: str, row: dict[str, Any]) -> str:
        return f"<!-- ops-report-memory-task:v1 slot={slot} publication={row['publication']['body_sha256']} report={row['report_url'].rsplit('/', 1)[1]} -->"

    def _work_body(self, slot: str, row: dict[str, Any]) -> str:
        return (
            self._work_marker(slot, row)
            + f"\n\n## 目的・背景\n\n完了公開レポート {row['report_url']} の観察を専用memoryへ反映する。\n"
            + f"\n## 対象範囲\n\n`{memory.DIRECTORY}/` のみ。branch `ops-report-memory/{slot}`、HEAD `{row['head']}`。\n"
            + "公開元とjournalを照合し、既存PRの内容・ローカル検証を確認して正式reviewと通常merge gateへ進める。\n"
            + "新規実装workerや別PRの作成は不要。HEAD変更・観察の上書き・直接mergeは対象外。\n"
            + "\n## 完了条件\n\n公開identity・限定path・HEAD・ローカル検証、親の正式review、通常gate、実merge/closeを確認する。\n"
            + "保守枠の通常merge停止を維持する。PR CIの有無は条件にしない。\n"
            + f"\n## 検証証拠\n\n{row['validation']['checks']} をHEADに対して実行済み。\n"
            + "\n## 実行根拠\n\nAIの提案として既存dispatcherの自律着手判断へ渡す。人間の要件合意や担当予約を作成した記録ではない。\n"
        )

    def _ensure_work_issue(self, slot: str, row: dict[str, Any], run_dir: Path) -> None:
        body = self._work_body(slot, row)
        fields = "number,url,state,body,author"
        if not row["work_issue"]:
            candidates = shell.gh_list(
                [
                    "issue",
                    "list",
                    "--state",
                    "all",
                    "--search",
                    f'"ops-report-memory-task:v1 slot={slot}" in:body',
                    "--json",
                    fields,
                ],
                limit=100,
                cwd=self.repo_root,
            )
            matches = [
                r
                for r in candidates
                if r["body"].split("\n", 1)[0] == self._work_marker(slot, row)
            ]
            if len(matches) > 1:
                raise ValueError("multiple memory work issues; reconcile explicitly")
            if matches:
                row["work_issue"] = matches[0]["url"]
            else:
                if row["work_issue_create_attempted"]:
                    raise ValueError(
                        "memory work issue creation unconfirmed; search again, never resend"
                    )
                path = run_dir / f"memory-task-{slot}.md"
                path.write_text(body, encoding="utf-8")
                row["work_issue_create_attempted"] = True
                self.save()
                row["work_issue"] = shell.gh(
                    [
                        "issue",
                        "create",
                        "--title",
                        f"運用レポート {slot} の専用memoryを正式reviewで反映する",
                        "--label",
                        "ai-proposed",
                        "--body-file",
                        str(path),
                    ],
                    cwd=self.repo_root,
                ).strip()
            self.save()  # response loss is recovered by the stable marker, not a new issue
        self._verify_work_issue(slot, row)

    def _verify_work_issue(self, slot: str, row: dict[str, Any]) -> None:
        issue_url = row["work_issue"]
        prefix = row["report_url"].rsplit("/", 1)[0] + "/"
        if (
            not isinstance(issue_url, str)
            or not issue_url.startswith(prefix)
            or not issue_url.removeprefix(prefix).isdigit()
        ):
            raise ValueError("memory work issue repository/URL mismatch")
        issue = shell.gh_json(
            [
                "issue",
                "view",
                issue_url.rsplit("/", 1)[1],
                "--json",
                "number,url,state,body,author",
            ],
            cwd=self.repo_root,
        )
        if (
            issue["url"] != issue_url
            or issue["state"] != "OPEN"
            or issue["author"]["login"] != row["publication"]["author"]
            or requirement_body(issue["body"])
            != requirement_body(self._work_body(slot, row))
        ):
            raise ValueError("memory work issue changed/closed; reconcile explicitly")

    def _pr_body(self, slot: str, row: dict[str, Any]) -> str:
        return (
            self._marker(slot, row)
            + f"\n\n{row['report_url']} の継続観察。\n"
            + f"変更は `{memory.DIRECTORY}/` のみ。検証: {row['validation']['checks']}。\n"
            + "既存dispatcherの自律判断・親の正式review・通常merge gateで進める。\n"
            + f"\nCloses #{row['work_issue'].rsplit('/', 1)[1]}\n"
        )

    def _link_issue(
        self, slot: str, row: dict[str, Any], pr: dict[str, Any], run_dir: Path
    ) -> None:
        body = self._pr_body(slot, row)
        if pr["body"] != body:
            if row["stage"] == "review_handoff":
                raise ValueError("memory PR handoff changed; reconcile explicitly")
            # Includes migration of old PRs; never change branch/HEAD/content.
            path = run_dir / f"memory-pr-{slot}.md"
            path.write_text(body, encoding="utf-8")
            shell.gh(
                ["pr", "edit", row["pr_url"], "--body-file", str(path)],
                cwd=self.repo_root,
            )
        if self._view(slot, f"ops-report-memory/{slot}", row)["body"] != body:
            raise ValueError("memory PR work issue link not confirmed")

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


def validate_for_gate(repo_root: Path, pr: dict[str, Any]) -> None:
    """Additional content validation inside the ordinary gate, never a merge path.

    Review/coverage/authorization/dependencies/deadline remain the gate's job.
    The same XDG journal is read without modifying it or acquiring schedule.lock.
    Atomic journal replacement prevents a partially published handoff being read.
    """
    from . import proposals
    from .collect import pin_base

    store = Store(repo_root)
    matches = [
        (slot, row)
        for slot, row in store.records.items()
        if row["pr_url"] == pr["html_url"]
    ]
    if len(matches) != 1:
        raise ValueError("memory PR has no unique publication journal")
    slot, row = matches[0]
    if (
        row["state"] != "pending"
        or row["stage"] != "review_handoff"
        or row["head"] != pr["head"]["sha"]
    ):
        raise ValueError("memory PR handoff/HEAD mismatch")
    expected = memory.apply_update(
        row["files"], row["update"], source=row["report_url"], today=date.today()
    )
    store._verify_commit(row, expected)
    if row["validation"] != store._validation(row, expected):
        raise ValueError("memory PR local validation mismatch")
    store._verify_work_issue(slot, row)
    current = store._view(slot, f"ops-report-memory/{slot}", row)
    if current["body"] != store._pr_body(slot, row) or pr["body"] != current["body"]:
        raise ValueError("memory PR closing link/identity mismatch")
    store._check_open(row, current, pin_base(repo_root, "origin/main").sha)
    report = shell.gh_json(
        [
            "issue",
            "view",
            row["report_url"].rsplit("/", 1)[1],
            "--json",
            "url,body,author",
        ],
        cwd=repo_root,
    )
    meta, _ = proposals.parse_report(report["body"])
    if (
        report["url"] != row["report_url"]
        or report["author"]["login"] != row["publication"]["author"]
        or meta.slot != slot
        or meta.status != "complete"
    ):
        raise ValueError("memory publication no longer confirmed")
