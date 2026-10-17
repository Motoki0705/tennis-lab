#!/usr/bin/env python3
"""Poll open agent-authored PRs, review them with Codex, and run the merge gate.

The rules (authorship, review, dedup, gate, auto-merge flag) are documented in
README.md next to this file. All data given to the reviewer is collected here
deterministically; the agent only reads and judges, and this script posts results.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shutil
import string
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from opsproc import CommandError, run, run_process_group  # noqa: E402
from review_policy import (  # noqa: E402
    CHANGES_REQUESTED_LABEL,
    ESCALATION_STREAK,
    NEEDS_HUMAN_LABEL,
    REVIEWER,
    AuthorConflictError,
    Check,
    Comment,
    GateDecision,
    GateInput,
    GateResult,
    Review,
    ReviewRecord,
    ReviewTarget,
    detect_author_agent,
    escalated_streaks,
    escalation_marker,
    evaluate_gate,
    gate_marker,
    gate_records,
    gate_should_comment,
    latest_review_for,
    load_list_file,
    load_shared_patterns,
    neutralize_markers,
    normalize_rollup,
    parse_review,
    request_changes_streak,
    review_marker,
    review_records,
)

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[2]
EXIT_LOCKED = 75
MAX_DIFF_CHARS = 200_000
MAX_COMMENT_CHARS = 60_000
TERM_GRACE_SECONDS = 10.0
# agents.conf profile pinning the reviewer's model / effort / service tier
# (REVIEW_CODEX_*), separate from the defaults other jobs use.
REVIEW_PROFILE = "review"
# Full diff written into the review worktree when the prompt copy is truncated (the
# read-only reviewer can read files but cannot run git).
FULL_DIFF_NAME = ".cross-review-full.diff"
PR_FIELDS = (
    "number,title,body,url,headRefName,headRefOid,baseRefName,baseRefOid,"
    "isDraft,isCrossRepository,labels,mergeable,statusCheckRollup"
)

log = logging.getLogger("cross_review")


def run_in_process_group(cmd: Sequence[str], *, cwd: Path, timeout: float) -> None:
    """Run the reviewer in its own process group (see ``opsproc.run_process_group``)."""
    run_process_group(cmd, cwd=cwd, timeout=timeout, term_grace=TERM_GRACE_SECONDS)


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
            Comment(
                author=str(item["user"]["login"]),
                body=str(item["body"] or ""),
                url=str(item["html_url"]),
            )
            for page in pages
            for item in page
        ]

    def _body_file(self, kind: str, number: int, body: str) -> Path:
        if len(body) > MAX_COMMENT_CHARS:
            raise ValueError(
                f"{kind} for #{number} is {len(body)} chars (> {MAX_COMMENT_CHARS})"
            )
        body_file = self.scratch / f"{kind}-{number}-{time.time_ns()}.md"
        body_file.write_text(body, encoding="utf-8")
        return body_file

    def comment(self, number: int, body: str) -> str:
        """Post a PR comment and return its URL (``gh pr comment`` prints it)."""
        out = run(
            [
                "gh", "pr", "comment", str(number),
                "--body-file", str(self._body_file("comment", number, body)),
            ],
            cwd=self.repo,
        )  # fmt: skip
        return _printed_url(out, f"gh pr comment {number}")

    def add_label(self, number: int, label: str) -> None:
        run(["gh", "pr", "edit", str(number), "--add-label", label], cwd=self.repo)

    def remove_label(self, number: int, label: str) -> None:
        run(["gh", "pr", "edit", str(number), "--remove-label", label], cwd=self.repo)

    def merge(self, number: int, sha: str) -> None:
        run(
            ["gh", "pr", "merge", str(number), "--merge", "--match-head-commit", sha],
            cwd=self.repo,
        )


def _printed_url(out: str, what: str) -> str:
    lines = [line for line in out.splitlines() if line.strip()]
    if not lines or not lines[-1].startswith("https://"):
        raise CommandError(f"{what} did not print a URL: {out!r}")
    return lines[-1].strip()


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

    def merge_base(self, pr: PullRequest) -> str:
        return self("merge-base", pr.base_sha, pr.head_sha).strip()

    def changed_paths(self, merge_base: str, head: str) -> tuple[str, ...]:
        # -z: raw NUL-separated names. Without it git quotes and escapes non-ASCII or
        # special paths, which would then never match the shared-path patterns.
        out = self("diff", "--name-only", "--no-renames", "-z", merge_base, head)
        if out and not out.endswith("\0"):
            raise CommandError(f"unexpected git diff -z output: {out[-200:]!r}")
        return tuple(out.split("\0")[:-1]) if out else ()

    def diff(self, merge_base: str, head: str) -> str:
        return self("diff", "--no-color", "--no-ext-diff", merge_base, head)

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
    agents_md: str,
    changed: Sequence[str],
    checks: Sequence[Check],
    diff: str,
) -> tuple[str, bool]:
    truncated = len(diff) > MAX_DIFF_CHARS
    diff_note = (
        f"diff は {len(diff)} 文字のため先頭 {MAX_DIFF_CHARS} 文字で切っています。"
        f"全文はカレントディレクトリの `{FULL_DIFF_NAME}`（呼び出し側が書き出したもので PR の一部ではない）"
        "をファイルとして読んでください。"
        if truncated
        else "diff 全体を示します。"
    )
    prompt = string.Template(settings.prompt_template).substitute(
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
    settings: Settings, *, cwd: Path, prompt: str, run_dir: Path
) -> Review:
    prompt_file = run_dir / "prompt.md"
    out_file = run_dir / "agent_output.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    run_in_process_group(
        [
            str(settings.agent_exec),
            "--agent", REVIEWER,
            "--profile", REVIEW_PROFILE,
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
    target: ReviewTarget,
    review: Review,
    *,
    author: str,
    author_source: str,
    elapsed: float,
    diff_truncated: bool,
) -> str:
    lines = [
        review_marker(target, REVIEWER, review.outcome),
        f"## 自動レビュー（{REVIEWER} → {author} 作成PR）",
        "",
        f"- 判定: **{review.verdict}**",
        f"- 対象 head: `{target.head}`（merge-base `{target.merge_base}`、base `{pr.base_ref}`）",
        f"- 作成 agent の判別根拠: {author_source}",
        f"- 所要時間: {elapsed:.0f} 秒",
    ]
    if diff_truncated:
        lines.append(
            f"- 注意: プロンプト内の diff は {MAX_DIFF_CHARS} 文字で切り、全文は worktree 内の"
            f" `{FULL_DIFF_NAME}` で読ませた"
        )
    # Agent text is untrusted: it must not be able to forge or hide a marker.
    inert = neutralize_markers
    lines += ["", "### 要約", "", inert(review.summary), "", "### 指摘", ""]
    if not review.findings:
        lines.append("（なし）")
    for finding in review.findings:
        where = finding.path or "(全体)"
        if finding.path and finding.line:
            where = f"{finding.path}:{finding.line}"
        lines.append(
            f"- **{finding.severity}** `{inert(where)}` ({inert(finding.rule)}): "
            f"{inert(finding.detail)}"
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
    "review_missing": "最新の対象への自動レビューが無い",
    "review_request_changes": "自動レビューが request_changes",
    "touches_shared": "共有基盤パスに触れている",
}


def format_gate_comment(
    target: ReviewTarget, result: GateResult, *, action: str
) -> str:
    lines = [
        gate_marker(target, result.decision.value, result.reasons),
        "## merge gate",
        "",
        f"- 判定: **{result.decision.value}**",
        f"- 対象 head: `{target.head}`（merge-base `{target.merge_base}`）",
        f"- 動作: {action}",
    ]
    if result.reasons:
        lines += ["", "### 理由", ""]
        lines += [f"- `{reason}`: {REASON_TEXT[reason]}" for reason in result.reasons]
    if result.shared_hits:
        lines += ["", "### 共有基盤に該当したファイル", ""]
        lines += [
            f"- `{neutralize_markers(path)}`（`{pattern}`）"
            for path, pattern in result.shared_hits
        ]
    lines += ["", "<sub>.agents/ops/review/cross_review.py による自動判定</sub>"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- per-PR flow


def review_head(
    settings: Settings,
    gh: GitHub,
    git: Git,
    pr: PullRequest,
    *,
    author: str,
    author_source: str,
    target: ReviewTarget,
    diff: str,
    changed: Sequence[str],
    checks: Sequence[Check],
    report: RunReport,
) -> ReviewRecord:
    """Run the reviewer on ``target``, post the result and return it as a record."""
    number = pr.number
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = settings.state_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Repeated reviews can have the same timestamp and target; preserve each one.
    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{stamp}-pr{number}-{target.head[:12]}-{target.merge_base[:12]}-",
            dir=runs_dir,
        )
    )
    prompt, truncated = build_prompt(
        settings,
        pr,
        author=author,
        author_source=author_source,
        agents_md=git.show(pr.base_sha, "AGENTS.md"),
        changed=changed,
        checks=checks,
        diff=diff,
    )
    started = time.monotonic()
    report.note(
        f"#{number}: reviewing head {pr.head_sha[:12]} with {REVIEWER} (author {author}); log {run_dir}"
    )
    with git.review_worktree(pr) as worktree:
        if truncated:
            write_new_file(worktree / FULL_DIFF_NAME, diff)
        review = run_reviewer(
            settings,
            cwd=worktree,
            prompt=prompt,
            run_dir=run_dir,
        )
    elapsed = time.monotonic() - started
    body = format_review_comment(
        pr,
        target,
        review,
        author=author,
        author_source=author_source,
        elapsed=elapsed,
        diff_truncated=truncated,
    )
    (run_dir / "review_comment.md").write_text(body, encoding="utf-8")
    report.note(f"#{number}: {REVIEWER} verdict={review.verdict} ({elapsed:.0f}s)")
    url = None
    if settings.dry_run:
        print(f"----- [dry-run] review comment for #{number} -----\n{body}")
    else:
        url = gh.comment(number, body)
    return ReviewRecord(
        target=target, reviewer=REVIEWER, outcome=review.outcome, url=url
    )


def write_new_file(path: Path, text: str) -> None:
    """Create ``path`` exclusively, refusing any existing entry including symlinks.

    The directory is a PR checkout, so the name may already exist as a file or as a
    (dangling) symlink pointing outside the worktree; never write through it.
    """
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    except FileExistsError as exc:
        raise CommandError(
            f"{path} already exists in the PR checkout (file or symlink); refusing to write"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def sync_review_label(
    settings: Settings, gh: GitHub, pr: PullRequest, verdict: str
) -> None:
    """Make the changes-requested label match the current target's verdict.

    Runs on every poll, not only right after a review, so a label update that failed
    once is retried instead of being lost behind the review dedup.
    """
    if settings.dry_run:
        return
    has_label = CHANGES_REQUESTED_LABEL in pr.labels
    if verdict == "request_changes" and not has_label:
        gh.add_label(pr.number, CHANGES_REQUESTED_LABEL)
    if verdict != "request_changes" and has_label:
        gh.remove_label(pr.number, CHANGES_REQUESTED_LABEL)


@dataclass(frozen=True)
class ReviewInput:
    """The diff of a PR and the identity a review of it is recorded under."""

    target: ReviewTarget
    diff: str
    changed: tuple[str, ...]


def review_input(git: Git, pr: PullRequest) -> ReviewInput:
    git.fetch_pr(pr)
    merge_base = git.merge_base(pr)
    diff = git.diff(merge_base, pr.head_sha)
    return ReviewInput(
        target=ReviewTarget(
            head=pr.head_sha,
            merge_base=merge_base,
            diff_sha256=hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        ),
        diff=diff,
        changed=git.changed_paths(merge_base, pr.head_sha),
    )


def refetch_for_gate(
    gh: GitHub,
    git: Git,
    pr: PullRequest,
    reviewed: ReviewTarget,
    report: RunReport,
) -> PullRequest | None:
    """Re-read the PR right before the gate; ``None`` when the review no longer applies.

    Used whether the verdict comes from a fresh review (minutes old) or an existing
    one: the PR may have been retargeted, pushed, relabelled or re-run CI since the
    first view. The base branch merely moving on keeps the merge-base and diff, so it
    is not stale. Contradicting author signals raise, as on the first view.
    """
    fresh = gh.view(pr.number)
    if detect_author_agent(fresh.head_ref, fresh.labels) is None:
        report.note(
            f"#{pr.number}: no longer an agent-authored PR "
            f"(branch={fresh.head_ref!r} labels={list(fresh.labels)}); "
            "gate deferred to the next poll"
        )
        return None
    if (fresh.head_sha, fresh.base_sha) == (pr.head_sha, pr.base_sha):
        return fresh
    current = review_input(git, fresh).target
    if current != reviewed:
        report.note(
            f"#{pr.number}: PR changed since its review target was computed "
            f"(head {reviewed.head[:12]}->{current.head[:12]}, "
            f"merge-base {reviewed.merge_base[:12]}->{current.merge_base[:12]}, "
            f"diff {reviewed.diff_sha256[:12]}->{current.diff_sha256[:12]}); "
            "gate deferred to the next poll"
        )
        return None
    return fresh


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
    comments = gh.comments(number)
    reviewed = review_input(git, pr)
    target = reviewed.target

    records = review_records(comments, trusted_login=login)
    current = latest_review_for(records, target=target, reviewer=REVIEWER)
    if current is not None:
        report.note(
            f"#{number}: head {pr.head_sha[:12]} (merge-base {target.merge_base[:12]}) "
            f"already reviewed by {REVIEWER}: {current.outcome.verdict}"
        )
    else:
        current = review_head(
            settings,
            gh,
            git,
            pr,
            author=author,
            author_source=author_source,
            target=target,
            diff=reviewed.diff,
            changed=reviewed.changed,
            checks=normalize_rollup(pr.rollup),
            report=report,
        )
        records = (*records, current)
    # Label and escalation are re-derived from the recorded reviews on every poll, so a
    # step that failed once is retried without reviewing again.
    sync_review_label(settings, gh, pr, current.outcome.verdict)
    escalate_streak(settings, gh, login, pr, records, comments, report)

    # Gate on the PR / CI state as it is now, for the exact target that was reviewed.
    fresh = refetch_for_gate(gh, git, pr, target, report)
    if fresh is None:
        return
    pr = fresh

    result = evaluate_gate(
        GateInput(
            is_draft=pr.is_draft,
            is_cross_repository=pr.is_cross_repository,
            mergeable=pr.mergeable,
            checks=normalize_rollup(pr.rollup),
            required_checks=settings.required_checks,
            review=current.outcome,
            changed_paths=reviewed.changed,
            shared_patterns=settings.shared_patterns,
        )
    )
    report.note(
        f"#{number}: gate={result.decision.value} reasons={list(result.reasons)} "
        f"shared={[path for path, _ in result.shared_hits]}"
    )
    apply_gate(settings, gh, login, pr, target, result, comments, report)


def escalate_streak(
    settings: Settings,
    gh: GitHub,
    login: str,
    pr: PullRequest,
    records: Sequence[ReviewRecord],
    comments: Sequence[Comment],
    report: RunReport,
) -> None:
    """Hand the PR to a human after ESCALATION_STREAK request_changes in a row.

    Once per streak (keyed by the streak's first review); a passing review starts over.
    """
    streak = request_changes_streak(records, reviewer=REVIEWER)
    if len(streak) < ESCALATION_STREAK:
        return
    start = streak[0].target
    if start in escalated_streaks(comments, trusted_login=login):
        return
    lines = [
        escalation_marker(start),
        "## 人間の判断が必要です",
        "",
        f"自動レビュー（{REVIEWER}）が {len(streak)} 回連続で request_changes を返しました。"
        f"自動の修正・再レビューでは収束していないため、ラベル `{NEEDS_HUMAN_LABEL}` を付けました。"
        "merge gate は request_changes のあいだ blocked のままです。",
        "",
        "### 連続した request_changes",
        "",
    ]
    lines += [
        f"- `{record.target.head[:12]}`: {record.url or '(dry-run)'}"
        for record in streak
    ]
    lines += ["", "<sub>.agents/ops/review/cross_review.py による自動判定</sub>"]
    body = "\n".join(lines) + "\n"
    report.note(f"#{pr.number}: escalated after {len(streak)} request_changes in a row")
    if settings.dry_run:
        print(f"----- [dry-run] escalation for #{pr.number} -----\n{body}")
        return
    # Label first: the comment is the record that the escalation is complete.
    if NEEDS_HUMAN_LABEL not in pr.labels:
        gh.add_label(pr.number, NEEDS_HUMAN_LABEL)
    gh.comment(pr.number, body)


def apply_gate(
    settings: Settings,
    gh: GitHub,
    login: str,
    pr: PullRequest,
    target: ReviewTarget,
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
        record.target == target
        and record.decision == result.decision.value
        and record.reasons == result.reasons
        for record in gate_records(comments, trusted_login=login)
    )
    body = format_gate_comment(target, result, action=action)
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
            required = {NEEDS_HUMAN_LABEL, CHANGES_REQUESTED_LABEL}
            missing = required - gh.label_names()
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
