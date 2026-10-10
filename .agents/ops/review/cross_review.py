#!/usr/bin/env python3
"""Poll open PRs, cross-review them with the other agent, and run the merge gate.

The rules (authorship, review, dedup, gate, auto-merge flag) are documented in
README.md next to this file. All data given to the reviewer is collected here
deterministically; the agent only reads and judges, and this script posts results.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import shutil
import string
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from review_policy import (  # noqa: E402
    CHANGES_REQUESTED_LABEL,
    NEEDS_HUMAN_LABEL,
    AuthorConflictError,
    Check,
    Comment,
    GateDecision,
    GateInput,
    GateResult,
    Review,
    detect_author_agent,
    evaluate_gate,
    gate_marker,
    gate_records,
    gate_should_comment,
    latest_review_for,
    load_list_file,
    load_shared_patterns,
    normalize_rollup,
    parse_review,
    review_marker,
    review_records,
    reviewer_for,
)

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[2]
EXIT_LOCKED = 75
MAX_DIFF_CHARS = 200_000
MAX_COMMENT_CHARS = 60_000
PR_FIELDS = (
    "number,title,body,url,headRefName,headRefOid,baseRefName,baseRefOid,"
    "isDraft,isCrossRepository,labels,mergeable,statusCheckRollup"
)

log = logging.getLogger("cross_review")


class CommandError(RuntimeError):
    pass


def run(cmd: Sequence[str], *, cwd: Path, timeout: float | None = None) -> str:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandError(f"timed out after {timeout}s: {' '.join(cmd)}") from exc
    if proc.returncode != 0:
        raise CommandError(
            f"exit {proc.returncode}: {' '.join(cmd)}\nstderr:\n{proc.stderr.strip()}"
        )
    return proc.stdout


# --------------------------------------------------------------------------- GitHub


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    body: str
    url: str
    head_ref: str
    head_sha: str
    base_ref: str
    base_sha: str
    is_draft: bool
    is_cross_repository: bool
    labels: tuple[str, ...]
    mergeable: str
    rollup: tuple[dict[str, Any], ...]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PullRequest:
        return cls(
            number=int(data["number"]),
            title=str(data["title"]),
            body=str(data["body"] or ""),
            url=str(data["url"]),
            head_ref=str(data["headRefName"]),
            head_sha=str(data["headRefOid"]),
            base_ref=str(data["baseRefName"]),
            base_sha=str(data["baseRefOid"]),
            is_draft=bool(data["isDraft"]),
            is_cross_repository=bool(data["isCrossRepository"]),
            labels=tuple(str(label["name"]) for label in data["labels"]),
            mergeable=str(data["mergeable"]),
            rollup=tuple(data["statusCheckRollup"] or ()),
        )


class GitHub:
    def __init__(self, repo: Path, scratch: Path) -> None:
        self.repo = repo
        self.scratch = scratch

    def _json(self, *args: str) -> Any:
        return json.loads(run(["gh", *args], cwd=self.repo))

    def login(self) -> str:
        return str(self._json("api", "user")["login"])

    def label_names(self) -> set[str]:
        return {
            str(item["name"])
            for item in self._json("label", "list", "--limit", "1000", "--json", "name")
        }

    def open_pr_numbers(self) -> list[int]:
        items = self._json(
            "pr", "list", "--state", "open", "--limit", "200", "--json", "number"
        )
        return sorted(int(item["number"]) for item in items)

    def view(self, number: int) -> PullRequest:
        return PullRequest.from_json(
            self._json("pr", "view", str(number), "--json", PR_FIELDS)
        )

    def comments(self, number: int) -> list[Comment]:
        pages = self._json(
            "api",
            "--paginate",
            "--slurp",
            f"repos/{{owner}}/{{repo}}/issues/{number}/comments",
        )
        return [
            Comment(author=str(item["user"]["login"]), body=str(item["body"] or ""))
            for page in pages
            for item in page
        ]

    def comment(self, number: int, body: str) -> None:
        if len(body) > MAX_COMMENT_CHARS:
            raise ValueError(
                f"comment for #{number} is {len(body)} chars (> {MAX_COMMENT_CHARS})"
            )
        body_file = self.scratch / f"comment-{number}-{time.time_ns()}.md"
        body_file.write_text(body, encoding="utf-8")
        run(
            ["gh", "pr", "comment", str(number), "--body-file", str(body_file)],
            cwd=self.repo,
        )

    def add_label(self, number: int, label: str) -> None:
        run(["gh", "pr", "edit", str(number), "--add-label", label], cwd=self.repo)

    def remove_label(self, number: int, label: str) -> None:
        run(["gh", "pr", "edit", str(number), "--remove-label", label], cwd=self.repo)

    def merge(self, number: int, sha: str) -> None:
        run(
            ["gh", "pr", "merge", str(number), "--merge", "--match-head-commit", sha],
            cwd=self.repo,
        )


# --------------------------------------------------------------------------- git


class Git:
    def __init__(self, repo: Path, remote: str) -> None:
        self.repo = repo
        self.remote = remote

    def __call__(self, *args: str) -> str:
        return run(["git", *args], cwd=self.repo)

    def fetch_pr(self, pr: PullRequest) -> None:
        self(
            "fetch",
            "--no-tags",
            "--quiet",
            self.remote,
            f"refs/heads/{pr.base_ref}",
            f"refs/pull/{pr.number}/head",
        )
        for sha in (pr.base_sha, pr.head_sha):
            try:
                self("cat-file", "-e", f"{sha}^{{commit}}")
            except CommandError as exc:
                raise CommandError(
                    f"#{pr.number}: commit {sha} not available after fetch (PR moved during poll?)"
                ) from exc

    def changed_paths(self, pr: PullRequest) -> tuple[str, ...]:
        out = self(
            "diff", "--name-only", "--no-renames", f"{pr.base_sha}...{pr.head_sha}"
        )
        return tuple(line for line in out.splitlines() if line)

    def diff(self, pr: PullRequest) -> str:
        return self(
            "diff", "--no-color", "--no-ext-diff", f"{pr.base_sha}...{pr.head_sha}"
        )

    def show(self, sha: str, path: str) -> str:
        return self("show", f"{sha}:{path}")

    @contextlib.contextmanager
    def review_worktree(self, pr: PullRequest) -> Iterator[Path]:
        path = (
            self.repo
            / ".claude"
            / "worktrees"
            / f"review-pr{pr.number}-{pr.head_sha[:12]}"
        )
        if path.exists():
            log.warning("#%d: removing stale review worktree %s", pr.number, path)
            self("worktree", "remove", "--force", str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        self("worktree", "add", "--detach", "--quiet", str(path), pr.head_sha)
        try:
            yield path
        finally:
            self("worktree", "remove", "--force", str(path))


# --------------------------------------------------------------------------- review


@dataclass
class Settings:
    repo: Path
    state_dir: Path
    agent_exec: Path
    agent_timeout: float
    shared_patterns: tuple[str, ...]
    required_checks: tuple[str, ...]
    prompt_template: str
    dry_run: bool
    auto_merge: bool


@dataclass
class RunReport:
    errors: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        log.info(message)
        self.lines.append(message)


def ci_summary(checks: Sequence[Check]) -> str:
    if not checks:
        return "(check なし)"
    return "\n".join(f"- {check.name}: {check.state.value}" for check in checks)


def build_prompt(
    settings: Settings,
    pr: PullRequest,
    *,
    author: str,
    author_source: str,
    reviewer: str,
    agents_md: str,
    changed: Sequence[str],
    checks: Sequence[Check],
    diff: str,
) -> tuple[str, bool]:
    truncated = len(diff) > MAX_DIFF_CHARS
    diff_note = (
        f"diff は {len(diff)} 文字のため先頭 {MAX_DIFF_CHARS} 文字で切っています。"
        f"残りは `git diff {pr.base_sha}...HEAD` で読んでください。"
        if truncated
        else "diff 全体を示します。"
    )
    prompt = string.Template(settings.prompt_template).substitute(
        reviewer=reviewer,
        author=author,
        author_source=author_source,
        pr_number=pr.number,
        url=pr.url,
        head_ref=pr.head_ref,
        base_ref=pr.base_ref,
        head_sha=pr.head_sha,
        base_sha=pr.base_sha,
        agents_md=agents_md.strip(),
        title=pr.title,
        body=pr.body.strip() or "(本文なし)",
        ci_summary=ci_summary(checks),
        changed_files="\n".join(f"- `{path}`" for path in changed) or "(なし)",
        diff_note=diff_note,
        diff=diff[:MAX_DIFF_CHARS].rstrip("\n"),
    )
    return prompt, truncated


def run_reviewer(
    settings: Settings, *, reviewer: str, cwd: Path, prompt: str, run_dir: Path
) -> Review:
    prompt_file = run_dir / "prompt.md"
    out_file = run_dir / "agent_output.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    run(
        [
            str(settings.agent_exec),
            "--agent", reviewer,
            "--cwd", str(cwd),
            "--prompt-file", str(prompt_file),
            "--out", str(out_file),
            "--mode", "read-only",
            "--log-dir", str(run_dir),
        ],
        cwd=settings.repo,
        timeout=settings.agent_timeout,
    )  # fmt: skip
    review = parse_review(out_file.read_text(encoding="utf-8"))
    (run_dir / "review.json").write_text(
        json.dumps(
            {
                "verdict": review.verdict,
                "summary": review.summary,
                "findings": [vars(finding) for finding in review.findings],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return review


def format_review_comment(
    pr: PullRequest,
    review: Review,
    *,
    author: str,
    author_source: str,
    reviewer: str,
    elapsed: float,
    diff_truncated: bool,
) -> str:
    lines = [
        review_marker(pr.head_sha, reviewer, review.verdict),
        f"## クロスレビュー（{reviewer} → {author} 作成PR）",
        "",
        f"- 判定: **{review.verdict}**",
        f"- 対象 head: `{pr.head_sha}`",
        f"- 作成 agent の判別根拠: {author_source}",
        f"- 所要時間: {elapsed:.0f} 秒",
    ]
    if diff_truncated:
        lines.append(
            f"- 注意: プロンプト内の diff は {MAX_DIFF_CHARS} 文字で切り、残りは worktree で読ませた"
        )
    lines += ["", "### 要約", "", review.summary, "", "### 指摘", ""]
    if not review.findings:
        lines.append("（なし）")
    for finding in review.findings:
        where = finding.path or "(全体)"
        if finding.path and finding.line:
            where = f"{finding.path}:{finding.line}"
        lines.append(
            f"- **{finding.severity}** `{where}` ({finding.rule}): {finding.detail}"
        )
    if review.verdict == "request_changes":
        lines += [
            "",
            f"ラベル `{CHANGES_REQUESTED_LABEL}` を付けました。作成 agent / 人間が修正を push すると、"
            "新しい head を再レビューします。",
        ]
    lines += ["", "<sub>.agents/ops/review/cross_review.py による自動レビュー</sub>"]
    return "\n".join(lines) + "\n"


REASON_TEXT = {
    "draft": "draft PR",
    "fork": "fork からの PR",
    "conflict": "base と conflict している",
    "mergeable_unknown": "GitHub が mergeable を計算中",
    "ci_failed": "失敗した CI check がある",
    "ci_pending": "CI が完了していない",
    "ci_missing_required": "必須 check（required_checks.txt）が未登録",
    "review_missing": "最新 head へのクロスレビューが無い",
    "review_request_changes": "クロスレビューが request_changes",
    "review_not_approved": "クロスレビューが comment（approve ではない）",
    "touches_shared": "共有基盤パスに触れている",
}


def format_gate_comment(pr: PullRequest, result: GateResult, *, action: str) -> str:
    lines = [
        gate_marker(pr.head_sha, result.decision.value, result.reasons),
        "## merge gate",
        "",
        f"- 判定: **{result.decision.value}**",
        f"- 対象 head: `{pr.head_sha}`",
        f"- 動作: {action}",
    ]
    if result.reasons:
        lines += ["", "### 理由", ""]
        lines += [f"- `{reason}`: {REASON_TEXT[reason]}" for reason in result.reasons]
    if result.shared_hits:
        lines += ["", "### 共有基盤に該当したファイル", ""]
        lines += [f"- `{path}`（`{pattern}`）" for path, pattern in result.shared_hits]
    lines += ["", "<sub>.agents/ops/review/cross_review.py による自動判定</sub>"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- per-PR flow


def process_pr(
    settings: Settings, gh: GitHub, git: Git, login: str, number: int, report: RunReport
) -> None:
    pr = gh.view(number)
    if pr.is_draft:
        report.note(f"#{number}: skip (draft)")
        return
    if pr.is_cross_repository:
        report.note(f"#{number}: skip (fork PR)")
        return
    author = detect_author_agent(pr.head_ref, pr.labels)
    if author is None:
        report.note(
            f"#{number}: skip (作成agentを判別できない: branch={pr.head_ref!r} labels={list(pr.labels)})"
        )
        return
    author_source = (
        f"branch `{pr.head_ref}`"
        if pr.head_ref.startswith(f"{author}/")
        else f"label `agent:{author}`"
    )
    reviewer = reviewer_for(author)
    comments = gh.comments(number)
    git.fetch_pr(pr)
    changed = git.changed_paths(pr)
    checks = normalize_rollup(pr.rollup)

    existing = latest_review_for(
        review_records(comments, trusted_login=login),
        sha=pr.head_sha,
        reviewer=reviewer,
    )
    if existing is not None:
        verdict = existing.verdict
        report.note(
            f"#{number}: head {pr.head_sha[:12]} already reviewed by {reviewer}: {verdict}"
        )
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = settings.state_dir / "runs" / f"{stamp}-pr{number}-{pr.head_sha[:12]}"
        run_dir.mkdir(parents=True)
        prompt, truncated = build_prompt(
            settings,
            pr,
            author=author,
            author_source=author_source,
            reviewer=reviewer,
            agents_md=git.show(pr.base_sha, "AGENTS.md"),
            changed=changed,
            checks=checks,
            diff=git.diff(pr),
        )
        started = time.monotonic()
        report.note(
            f"#{number}: reviewing head {pr.head_sha[:12]} with {reviewer} (author {author}); log {run_dir}"
        )
        with git.review_worktree(pr) as worktree:
            review = run_reviewer(
                settings,
                reviewer=reviewer,
                cwd=worktree,
                prompt=prompt,
                run_dir=run_dir,
            )
        elapsed = time.monotonic() - started
        body = format_review_comment(
            pr,
            review,
            author=author,
            author_source=author_source,
            reviewer=reviewer,
            elapsed=elapsed,
            diff_truncated=truncated,
        )
        (run_dir / "review_comment.md").write_text(body, encoding="utf-8")
        verdict = review.verdict
        report.note(f"#{number}: {reviewer} verdict={verdict} ({elapsed:.0f}s)")
        if settings.dry_run:
            print(f"----- [dry-run] review comment for #{number} -----\n{body}")
        else:
            gh.comment(number, body)
            if (
                verdict == "request_changes"
                and CHANGES_REQUESTED_LABEL not in pr.labels
            ):
                gh.add_label(number, CHANGES_REQUESTED_LABEL)
            if verdict != "request_changes" and CHANGES_REQUESTED_LABEL in pr.labels:
                gh.remove_label(number, CHANGES_REQUESTED_LABEL)

    result = evaluate_gate(
        GateInput(
            is_draft=pr.is_draft,
            is_cross_repository=pr.is_cross_repository,
            mergeable=pr.mergeable,
            checks=checks,
            required_checks=settings.required_checks,
            review_verdict=verdict,
            changed_paths=changed,
            shared_patterns=settings.shared_patterns,
        )
    )
    report.note(
        f"#{number}: gate={result.decision.value} reasons={list(result.reasons)} "
        f"shared={[path for path, _ in result.shared_hits]}"
    )
    apply_gate(settings, gh, login, pr, result, comments, report)


def apply_gate(
    settings: Settings,
    gh: GitHub,
    login: str,
    pr: PullRequest,
    result: GateResult,
    comments: Sequence[Comment],
    report: RunReport,
) -> None:
    if result.decision is GateDecision.MERGE:
        if settings.dry_run:
            action = "dry-run のため merge しない"
        elif settings.auto_merge:
            action = "AGENT_AUTO_MERGE=1 により merge commit で merge する"
        else:
            action = (
                "自動mergeは無効（AGENT_AUTO_MERGE 未設定）。人間が merge してよい状態"
            )
    elif result.decision is GateDecision.NEEDS_HUMAN:
        action = f"ラベル `{NEEDS_HUMAN_LABEL}` を付けて人間の承認を待つ"
    else:
        action = "merge しない"

    already = any(
        record.sha == pr.head_sha
        and record.decision == result.decision.value
        and record.reasons == result.reasons
        for record in gate_records(comments, trusted_login=login)
    )
    body = format_gate_comment(pr, result, action=action)
    if settings.dry_run:
        print(
            f"----- [dry-run] gate for #{pr.number} (would comment: {gate_should_comment(result) and not already}) -----\n{body}"
        )
        return
    if gate_should_comment(result) and not already:
        gh.comment(pr.number, body)
    if (
        result.decision is GateDecision.NEEDS_HUMAN
        and NEEDS_HUMAN_LABEL not in pr.labels
    ):
        gh.add_label(pr.number, NEEDS_HUMAN_LABEL)
    if result.decision is GateDecision.MERGE and settings.auto_merge:
        gh.merge(pr.number, pr.head_sha)
        report.note(f"#{pr.number}: merged {pr.head_sha[:12]}")


# --------------------------------------------------------------------------- entry point


def parse_auto_merge(raw: str | None) -> bool:
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"AGENT_AUTO_MERGE must be unset, '0' or '1', got {raw!r}")


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=DEFAULT_REPO, help="checkout used for gh/git"
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument(
        "--pr", type=int, action="append", help="only these PR numbers (repeatable)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="no comments, labels or merges"
    )
    parser.add_argument(
        "--state-dir", type=Path, default=state_home / "tennis-lab-agents/review"
    )
    parser.add_argument(
        "--agent-exec", type=Path, default=HERE.parent / "lib/agent_exec.sh"
    )
    parser.add_argument(
        "--agent-timeout", type=float, default=1800.0, help="seconds per review"
    )
    parser.add_argument("--shared-paths", type=Path, default=HERE / "shared_paths.txt")
    parser.add_argument(
        "--required-checks", type=Path, default=HERE / "required_checks.txt"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    try:
        auto_merge = parse_auto_merge(os.environ.get("AGENT_AUTO_MERGE"))
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    repo = args.repo.resolve()
    settings = Settings(
        repo=repo,
        state_dir=args.state_dir.resolve(),
        agent_exec=args.agent_exec.resolve(),
        agent_timeout=args.agent_timeout,
        shared_patterns=load_shared_patterns(args.shared_paths),
        required_checks=load_list_file(args.required_checks),
        prompt_template=(HERE / "prompt_template.md").read_text(encoding="utf-8"),
        dry_run=args.dry_run,
        auto_merge=auto_merge,
    )
    if not os.access(settings.agent_exec, os.X_OK):
        log.error("agent_exec is not executable: %s", settings.agent_exec)
        return 2

    with exclusive_lock(settings.state_dir / "poll.lock") as acquired:
        if not acquired:
            log.warning(
                "another cross_review run holds %s; exiting",
                settings.state_dir / "poll.lock",
            )
            return EXIT_LOCKED
        scratch = settings.state_dir / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        gh = GitHub(repo, scratch)
        git = Git(repo, args.remote)
        login = gh.login()
        if not settings.dry_run:
            missing = {NEEDS_HUMAN_LABEL, CHANGES_REQUESTED_LABEL} - gh.label_names()
            if missing:
                log.error(
                    "required labels missing in the repo (see README): %s",
                    sorted(missing),
                )
                return 2
        numbers = sorted(set(args.pr)) if args.pr else gh.open_pr_numbers()
        report = RunReport()
        log.info(
            "poll start: prs=%s dry_run=%s auto_merge=%s login=%s",
            numbers, settings.dry_run, settings.auto_merge, login,
        )  # fmt: skip
        for number in numbers:
            try:
                process_pr(settings, gh, git, login, number, report)
            except (CommandError, AuthorConflictError, ValueError, OSError) as exc:
                log.error("#%d: %s", number, exc)
                report.errors.append(f"#{number}: {exc}")
        shutil.rmtree(scratch)
    if report.errors:
        log.error("poll finished with %d error(s)", len(report.errors))
        return 1
    log.info("poll finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
