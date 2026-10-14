#!/usr/bin/env python3
"""Resolve cross-review follow-up issues: close them, or implement and open a PR.

The rules (targets, rounds, agent contract, publishing) are documented in README.md
next to this file. This script selects issues, prepares a worktree, runs the agent
(``agent_exec.sh --profile followup --mode write``), validates its output and the
branch, and does every GitHub write itself.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import string
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE.parent / "review"))

from followup_policy import (  # noqa: E402
    TRUSTED_ASSOCIATIONS,
    IssueComment,
    NextStep,
    Resolution,
    ResolutionParseError,
    Round,
    SourceAction,
    branch_name,
    check_commits,
    disposition_table,
    is_followup_branch,
    next_step,
    parse_resolution,
    recorded_rounds,
    rejected_marker,
    rejection_noted,
    round_marker,
    source_action,
)
from opsproc import CommandError, run, run_process_group  # noqa: E402
from review_policy import (  # noqa: E402
    FOLLOWUP_LABEL,
    NEEDS_HUMAN_LABEL,
    Finding,
    Issue,
    followup_issues,
    neutralize_markers,
    parse_followup_data,
)

DEFAULT_REPO = HERE.parents[2]
EXIT_LOCKED = 75
# Environment the write-mode agent must not inherit: it may edit and commit, but
# pushing, opening PRs and touching issues are this script's job.
SCRUBBED_ENV = ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "SSH_AUTH_SOCK")
PROFILE = "followup"
AGENT = "codex"
MAX_SOURCE_BODY_CHARS = 20_000

log = logging.getLogger("resolve_followups")


# --------------------------------------------------------------------------- GitHub


@dataclass(frozen=True)
class Candidate:
    issue: Issue
    source_pr: int
    source_head: str
    labels: frozenset[str]


@dataclass(frozen=True)
class SourcePR:
    number: int
    state: str
    title: str
    url: str
    head_ref: str
    merge_commit: str | None
    body: str


class GitHub:
    def __init__(self, repo: Path, scratch: Path) -> None:
        self.repo = repo
        self.scratch = scratch

    def _json(self, *args: str) -> Any:
        return json.loads(run(["gh", *args], cwd=self.repo))

    def login(self) -> str:
        return str(self._json("api", "user")["login"])

    def label_names(self) -> set[str]:
        rows = self._json("label", "list", "--limit", "500", "--json", "name")
        return {str(r["name"]) for r in rows}

    def open_followups(self, author: str) -> list[Candidate]:
        pages = self._json(
            "api",
            "--paginate",
            "--slurp",
            f"repos/{{owner}}/{{repo}}/issues?labels={FOLLOWUP_LABEL}"
            f"&state=open&creator={author}&per_page=100",
        )
        items = [i for page in pages for i in page if "pull_request" not in i]
        issues = [
            Issue(
                number=int(i["number"]),
                author=str(i["user"]["login"]),
                body=str(i["body"] or ""),
                url=str(i["html_url"]),
            )
            for i in items
        ]
        labels = {
            int(i["number"]): frozenset(lb["name"] for lb in i["labels"]) for i in items
        }
        filed = followup_issues(issues, trusted_login=author)
        return sorted(
            (
                Candidate(issue, pr, head, labels[issue.number])
                for (pr, head), issue in filed.items()
            ),
            key=lambda c: c.issue.number,
        )

    def source_pr(self, number: int) -> SourcePR:
        data = self._json(
            "pr",
            "view",
            str(number),
            "--json",
            "number,state,title,url,headRefName,mergeCommit,body",
        )
        merge = data.get("mergeCommit")
        return SourcePR(
            number=int(data["number"]),
            state=str(data["state"]),
            title=str(data["title"]),
            url=str(data["url"]),
            head_ref=str(data["headRefName"]),
            merge_commit=str(merge["oid"]) if merge else None,
            body=str(data["body"] or ""),
        )

    def followup(self, number: int, author: str) -> Candidate | None:
        """Read current eligibility and input instead of trusting the initial list."""
        data = self._json("api", f"repos/{{owner}}/{{repo}}/issues/{number}")
        labels = frozenset(label["name"] for label in data["labels"])
        if data["state"] != "open" or "pull_request" in data or FOLLOWUP_LABEL not in labels:
            return None
        issue = Issue(
            number=int(data["number"]),
            author=str(data["user"]["login"]),
            body=str(data["body"] or ""),
            url=str(data["html_url"]),
        )
        filed = followup_issues([issue], trusted_login=author)
        if not filed:
            return None
        ((pr, head),) = filed
        return Candidate(issue, pr, head, labels)

    def pr_state(self, number: int) -> str:
        return str(self._json("pr", "view", str(number), "--json", "state")["state"])

    def comments(self, issue: int) -> list[IssueComment]:
        pages = self._json(
            "api",
            "--paginate",
            "--slurp",
            f"repos/{{owner}}/{{repo}}/issues/{issue}/comments",
        )
        return [
            IssueComment(
                author=str(c["user"]["login"]),
                association=str(c["author_association"]),
                body=str(c["body"] or ""),
                url=str(c["html_url"]),
            )
            for page in pages
            for c in page
        ]

    def _body_file(self, kind: str, number: int, body: str) -> Path:
        path = self.scratch / f"{kind}-{number}-{time.time_ns()}.md"
        path.write_text(body, encoding="utf-8")
        return path

    def comment(self, issue: int, body: str) -> str:
        out = run(
            ["gh", "issue", "comment", str(issue),
             "--body-file", str(self._body_file("comment", issue, body))],
            cwd=self.repo,
        )  # fmt: skip
        return _printed_url(out, f"gh issue comment {issue}")

    def add_label(self, issue: int, label: str) -> None:
        run(["gh", "issue", "edit", str(issue), "--add-label", label], cwd=self.repo)

    def close(self, issue: int, reason: str) -> None:
        run(["gh", "issue", "close", str(issue), "--reason", reason], cwd=self.repo)

    def create_pr(self, *, base: str, head: str, title: str, body: str) -> str:
        out = run(
            ["gh", "pr", "create", "--base", base, "--head", head, "--title", title,
             "--body-file", str(self._body_file("pr", 0, body))],
            cwd=self.repo,
        )  # fmt: skip
        return _printed_url(out, f"gh pr create --head {head}")


def _printed_url(out: str, what: str) -> str:
    lines = [ln for ln in out.strip().splitlines() if ln.startswith("https://")]
    if not lines:
        raise CommandError(f"{what} did not print a URL: {out!r}")
    return lines[-1]


def _pr_number(url: str) -> int:
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    if not tail.isdigit():
        raise CommandError(f"not a PR URL: {url!r}")
    return int(tail)


# --------------------------------------------------------------------------- git


class Git:
    def __init__(self, repo: Path, remote: str) -> None:
        self.repo = repo
        self.remote = remote

    def __call__(self, *args: str, cwd: Path | None = None) -> str:
        return run(["git", *args], cwd=cwd or self.repo)

    def fetch_base(self, branch: str) -> str:
        self("fetch", "--quiet", self.remote, branch)
        return self(
            "rev-parse", "--verify", f"refs/remotes/{self.remote}/{branch}^{{commit}}"
        ).strip()

    def contains_commit(self, base: str, commit: str) -> bool:
        # A source PR may have merged into a branch other than our processing base.
        # Fetch its commit explicitly: fetching main need not have brought it in.
        self("fetch", "--quiet", self.remote, commit)
        return not self("rev-list", "--max-count=1", f"{base}..{commit}").strip()

    def remote_branch_exists(self, branch: str) -> bool:
        return bool(
            self("ls-remote", "--heads", self.remote, f"refs/heads/{branch}").strip()
        )

    def local_branch_exists(self, branch: str) -> bool:
        return bool(
            self("for-each-ref", "--format=%(refname)", f"refs/heads/{branch}").strip()
        )

    @contextlib.contextmanager
    def worktree(self, branch: str, base: str) -> Iterator[Path]:
        """A new branch at ``base`` checked out in ``.claude/worktrees/<name>``.

        Leftovers of an earlier run are an error, never reused: they may hold work
        nobody has looked at.
        """
        path = self.repo / ".claude/worktrees" / branch.replace("/", "-")
        if path.exists() or os.path.lexists(path):
            raise CommandError(f"worktree path already exists: {path}")
        if self.local_branch_exists(branch):
            raise CommandError(f"local branch already exists: {branch}")
        self("worktree", "add", "--quiet", "-b", branch, str(path), base)
        try:
            yield path
        finally:
            self("worktree", "remove", "--force", str(path))
            self("branch", "-D", branch)

    def branch_state(self, worktree: Path, branch: str, base: str) -> int:
        """Validate the agent's branch and return its commit count above ``base``."""
        current = self(
            "symbolic-ref", "--quiet", "--short", "HEAD", cwd=worktree
        ).strip()
        if current != branch:
            raise CommandError(
                f"agent left the worktree on {current!r}, expected {branch!r}"
            )
        dirty = self("status", "--porcelain=v1", "--untracked-files=all", cwd=worktree)
        if dirty.strip():
            raise CommandError(f"agent left uncommitted changes:\n{dirty.rstrip()}")
        run(["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=worktree)
        return int(self("rev-list", "--count", f"{base}..HEAD", cwd=worktree).strip())

    def push(self, branch: str) -> None:
        self("push", "--quiet", self.remote, f"refs/heads/{branch}:refs/heads/{branch}")


# --------------------------------------------------------------------------- run


@dataclass(frozen=True)
class Settings:
    repo: Path
    remote: str
    base_branch: str
    agent_exec: Path
    agent_timeout: float
    state_dir: Path
    dry_run: bool
    only: frozenset[int]


@dataclass
class RunReport:
    lines: list[str] = field(default_factory=list)
    errors: int = 0

    def note(self, message: str) -> None:
        log.info("%s", message)
        self.lines.append(message)

    def error(self, message: str) -> None:
        log.error("%s", message)
        self.lines.append(f"ERROR {message}")
        self.errors += 1


def finding_label(f: Finding) -> str:
    where = f.path or "(パスなし)"
    if f.line is not None:
        where = f"{where}:{f.line}"
    return f"{f.severity} {where} ({f.rule})"


def format_findings(findings: Sequence[Finding]) -> str:
    blocks = []
    for i, f in enumerate(findings, start=1):
        blocks.append(
            f"{i}. **{finding_label(f)}**\n\n   {neutralize_markers(f.detail)}"
        )
    return "\n\n".join(blocks)


def format_comments(comments: Sequence[IssueComment]) -> str:
    trusted = [c for c in comments if c.association in TRUSTED_ASSOCIATIONS]
    if not trusted:
        return "（なし）"
    return "\n\n".join(
        f"---\n**{c.author}**（{c.url}）:\n\n{neutralize_markers(c.body)}"
        for c in trusted
    )


def build_prompt(
    *,
    candidate: Candidate,
    source: SourcePR,
    findings: Sequence[Finding],
    comments: Sequence[IssueComment],
    rounds: Sequence[Round],
    settings: Settings,
    branch: str,
    base: str,
) -> str:
    template = string.Template(
        (HERE / "prompt_template.md").read_text(encoding="utf-8")
    )
    generation_note = (
        "\n**元 PR 自体が追補 PR です。** 指摘の連鎖を止めるため、正しさ（バグ・静かなフォールバック・"
        "テストの欠落）に関わるものだけを fixed にし、体裁や好みの nit は declined にしてください。\n"
        if is_followup_branch(source.head_ref)
        else ""
    )
    previous = ""
    if rounds:
        lines = [
            f"- 第 {r.number} 回: " + (f"PR #{r.pr}" if r.pr else "PR なし")
            for r in rounds
        ]
        previous = (
            "- これまでの処理（詳細は下のコメント。前回の PR が merge されずに close された場合は、"
            "その理由をコメントから読み取ってください）:\n" + "\n".join(lines) + "\n"
        )
    return template.substitute(
        branch=branch,
        base_ref=f"{settings.remote}/{settings.base_branch}",
        base=base,
        python=str(settings.repo / ".venv/bin/python"),
        source_head=candidate.source_head,
        generation_note=generation_note,
        source_number=source.number,
        source_title=neutralize_markers(source.title),
        source_url=source.url,
        source_merge_commit=source.merge_commit or "(不明)",
        source_body=source_body(source.body),
        issue_number=candidate.issue.number,
        issue_url=candidate.issue.url,
        round_no=len(rounds) + 1,
        previous_rounds=previous,
        findings=format_findings(findings),
        comments=format_comments(comments),
        n_findings=len(findings),
    )


def source_body(body: str) -> str:
    if not body.strip():
        return "（本文なし）"
    text = neutralize_markers(body)
    if len(text) > MAX_SOURCE_BODY_CHARS:
        text = (
            text[:MAX_SOURCE_BODY_CHARS]
            + f"\n\n（{MAX_SOURCE_BODY_CHARS} 文字で切り詰め）"
        )
    return text


def agent_env(run_dir: Path) -> dict[str, str]:
    """The agent's environment: no GitHub credentials and no way to push.

    ``gh`` sees an empty config dir, and git cannot reach the SSH remote. This is
    defence in depth behind the prompt; the write-mode agent is otherwise unsandboxed.
    """
    gh_dir = run_dir / "gh-config-empty"
    gh_dir.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV}
    env.update(
        GH_CONFIG_DIR=str(gh_dir),
        GIT_SSH_COMMAND="false",
        GIT_TERMINAL_PROMPT="0",
        GH_PROMPT_DISABLED="1",
    )
    return env


def run_agent(settings: Settings, *, cwd: Path, prompt: str, run_dir: Path) -> str:
    prompt_file = run_dir / "prompt.md"
    out_file = run_dir / "agent_output.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    run_process_group(
        [
            str(settings.agent_exec),
            "--agent", AGENT,
            "--profile", PROFILE,
            "--cwd", str(cwd),
            "--prompt-file", str(prompt_file),
            "--out", str(out_file),
            "--mode", "write",
            "--log-dir", str(run_dir),
        ],
        cwd=settings.repo,
        timeout=settings.agent_timeout,
        env=agent_env(run_dir),
    )  # fmt: skip
    return out_file.read_text(encoding="utf-8")


FOOTER = "<sub>.agents/ops/followup/resolve_followups.py による自動処理</sub>"


def resolution_comment(
    r: Round, resolution: Resolution, labels: Sequence[str], *, pr_url: str | None
) -> str:
    if pr_url and resolution.closes_issue:
        outcome = (
            f"追補 PR を作成しました: {pr_url}（merge でこの issue は close されます）"
        )
    elif pr_url:
        outcome = (
            f"対応できる指摘の PR を作成しました: {pr_url}。残りは人間の判断が必要なので "
            f"`{NEEDS_HUMAN_LABEL}` を付けます（判断をコメントしてラベルを外すと、次回の処理で再開します）"
        )
    elif resolution.needs_human:
        outcome = (
            f"人間の判断が必要な指摘があるので `{NEEDS_HUMAN_LABEL}` を付けます"
            "（判断をコメントしてラベルを外すと、次回の処理で再開します）"
        )
    else:
        outcome = "対応が必要な指摘は無いので close します"
    return "\n".join(
        [
            round_marker(r),
            f"### 追補 issue の処理（第 {r.number} 回、base `{r.base[:12]}`）",
            "",
            outcome,
            "",
            neutralize_markers(resolution.summary),
            "",
            disposition_table(resolution, labels),
            "",
            "<details><summary>確認内容</summary>",
            "",
            neutralize_markers(resolution.verification),
            "",
            "</details>",
            "",
            FOOTER,
        ]
    )


def pr_body(
    candidate: Candidate,
    source: SourcePR,
    resolution: Resolution,
    labels: Sequence[str],
) -> str:
    assert resolution.pr_body is not None
    link = "Closes" if resolution.closes_issue else "Refs"
    return "\n".join(
        [
            f"<!-- agent-followup-pr:v1 issue={candidate.issue.number} -->",
            neutralize_markers(resolution.pr_body),
            "",
            f"## 追補 issue #{candidate.issue.number} の指摘への処置",
            "",
            f"元 PR: #{source.number}",
            "",
            disposition_table(resolution, labels),
            "",
            "## 確認",
            "",
            neutralize_markers(resolution.verification),
            "",
            f"{link} #{candidate.issue.number}",
            "",
            FOOTER,
        ]
    )


def invalidate(
    settings: Settings, gh: GitHub, candidate: Candidate, source: SourcePR,
    rounds: Sequence[Round], base: str, report: RunReport,
) -> None:  # fmt: skip
    r = Round(candidate.issue.number, len(rounds) + 1, base, None)
    body = "\n".join(
        [
            round_marker(r),
            f"元 PR #{source.number} は merge されずに close されたので、この指摘は対象のコードと"
            "ともに無効になりました。close します。",
            "",
            FOOTER,
        ]
    )
    if settings.dry_run:
        report.note(
            f"#{candidate.issue.number}: would close (source PR closed, dry-run)"
        )
        return
    gh.comment(candidate.issue.number, body)
    gh.close(candidate.issue.number, "not planned")
    report.note(
        f"#{candidate.issue.number}: closed (source PR #{source.number} closed unmerged)"
    )


def note_rejection(
    settings: Settings,
    gh: GitHub,
    candidate: Candidate,
    latest: Round,
    report: RunReport,
) -> None:
    assert latest.pr is not None
    number = candidate.issue.number
    body = "\n".join(
        [
            rejected_marker(number, latest.number, latest.pr),
            f"追補 PR #{latest.pr} は merge されずに close されました。`{NEEDS_HUMAN_LABEL}` を付けます。",
            "この issue を close するか、方針をコメントしてラベルを外してください"
            "（外すと、次回の処理でコメントを踏まえて再度取り組みます）。",
            "",
            FOOTER,
        ]
    )
    if settings.dry_run:
        report.note(
            f"#{number}: would hand to a human (PR #{latest.pr} rejected, dry-run)"
        )
        return
    gh.add_label(number, NEEDS_HUMAN_LABEL)
    gh.comment(number, body)
    report.note(
        f"#{number}: PR #{latest.pr} was closed unmerged; labelled {NEEDS_HUMAN_LABEL}"
    )


def current_candidate(
    gh: GitHub, number: int, login: str, report: RunReport,
) -> Candidate | None:  # fmt: skip
    candidate = gh.followup(number, login)
    if candidate is None:
        report.note(f"#{number}: skip (no longer an open follow-up issue)")
        return None
    if NEEDS_HUMAN_LABEL in candidate.labels:
        report.note(f"#{number}: skip ({NEEDS_HUMAN_LABEL})")
        return None
    return candidate


def inputs_unchanged(
    gh: GitHub, login: str, candidate: Candidate, source: SourcePR,
    comments: Sequence[IssueComment], report: RunReport,
    *, latest: Round | None = None, latest_state: str | None = None,
) -> bool:  # fmt: skip
    """Stop before any publication if the decision's input or eligibility changed."""
    number = candidate.issue.number
    current = current_candidate(gh, number, login, report)
    if current is None:
        return False
    if current != candidate or gh.source_pr(candidate.source_pr) != source:
        report.note(f"#{number}: defer (issue or source PR changed during processing)")
        return False
    trusted_before = [c for c in comments if c.association in TRUSTED_ASSOCIATIONS]
    trusted_now = [c for c in gh.comments(number) if c.association in TRUSTED_ASSOCIATIONS]
    if trusted_now != trusted_before:
        report.note(f"#{number}: defer (trusted comments changed during processing)")
        return False
    if latest and latest.pr and gh.pr_state(latest.pr) != latest_state:
        report.note(f"#{number}: defer (previous PR #{latest.pr} changed during processing)")
        return False
    return True


def resolve_issue(
    settings: Settings, gh: GitHub, git: Git, login: str, candidate: Candidate,
    report: RunReport,
) -> None:  # fmt: skip
    number = candidate.issue.number
    refreshed = current_candidate(gh, number, login, report)
    if refreshed is None:
        return
    candidate = refreshed
    source = gh.source_pr(candidate.source_pr)
    action = source_action(source.state)
    if action is SourceAction.DEFER:
        report.note(f"#{number}: defer (source PR #{source.number} is still open)")
        return

    comments = gh.comments(number)
    rounds = recorded_rounds(number, comments, trusted_login=login)
    base = git.fetch_base(settings.base_branch)
    if action is SourceAction.INVALIDATE:
        if inputs_unchanged(gh, login, candidate, source, comments, report):
            invalidate(settings, gh, candidate, source, rounds, base, report)
        return

    if source.merge_commit is None:
        raise ValueError(f"#{number}: merged source PR #{source.number} has no merge commit")
    if not git.contains_commit(base, source.merge_commit):
        report.note(
            f"#{number}: defer (source PR #{source.number} merge commit is not in {settings.base_branch})"
        )
        return

    latest = rounds[-1] if rounds else None
    latest_state = gh.pr_state(latest.pr) if latest and latest.pr else None
    noted = latest is not None and rejection_noted(
        number, latest, comments, trusted_login=login
    )
    step = next_step(latest, latest_state, rejection_noted=noted)
    if step is NextStep.WAIT_REVIEW:
        assert latest is not None
        report.note(f"#{number}: skip (PR #{latest.pr} is in review)")
        return
    if step is NextStep.NOTE_REJECTION:
        assert latest is not None
        if inputs_unchanged(
            gh, login, candidate, source, comments, report,
            latest=latest, latest_state=latest_state,
        ):
            note_rejection(settings, gh, candidate, latest, report)
        return

    findings = parse_followup_data(candidate.issue).findings
    if not findings:
        raise ValueError(f"#{number}: follow-up issue lists no findings")
    round_no = len(rounds) + 1
    branch = branch_name(number, round_no)
    if git.remote_branch_exists(branch):
        raise CommandError(
            f"#{number}: remote branch {branch} exists without a recorded round "
            "(an earlier run pushed but did not finish); open its PR or delete it"
        )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = settings.state_dir / "runs" / f"{stamp}-i{number}-r{round_no}"
    run_dir.mkdir(parents=True)
    prompt = build_prompt(
        candidate=candidate, source=source, findings=findings, comments=comments,
        rounds=rounds, settings=settings, branch=branch, base=base,
    )  # fmt: skip
    labels = [finding_label(f) for f in findings]
    with git.worktree(branch, base) as worktree:
        report.note(
            f"#{number}: round {round_no} on {branch} (base {base[:12]}); agent running"
        )
        started = time.monotonic()
        output = run_agent(settings, cwd=worktree, prompt=prompt, run_dir=run_dir)
        report.note(f"#{number}: agent finished in {time.monotonic() - started:.0f}s")
        resolution = parse_resolution(output, len(findings))
        n_commits = git.branch_state(worktree, branch, base)
        check_commits(resolution, n_commits)
        (run_dir / "resolution.json").write_text(
            json.dumps(vars(resolution) | {"dispositions": [vars(d) for d in resolution.dispositions]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )  # fmt: skip
        if n_commits:
            (run_dir / "changes.diff").write_text(
                git("diff", f"{base}..HEAD", cwd=worktree), encoding="utf-8"
            )
        if inputs_unchanged(
            gh, login, candidate, source, comments, report,
            latest=latest, latest_state=latest_state,
        ):
            publish(settings, gh, git, candidate, source, resolution, labels,
                    Round(number, round_no, base, None), branch, report)  # fmt: skip


def publish(
    settings: Settings, gh: GitHub, git: Git, candidate: Candidate, source: SourcePR,
    resolution: Resolution, labels: Sequence[str], r: Round, branch: str, report: RunReport,
) -> None:  # fmt: skip
    """Push / open the PR, then record the round on the issue (comment first line)."""
    number = candidate.issue.number
    counts = {
        d: resolution.count(d)
        for d in ("fixed", "declined", "already_resolved", "needs_human")
    }
    if settings.dry_run:
        report.note(f"#{number}: would publish {counts} (dry-run)")
        return
    # Establish the human hold before recording a round or publishing a branch.
    # A failed label call must never look like a human deliberately removed it.
    if resolution.needs_human:
        gh.add_label(number, NEEDS_HUMAN_LABEL)
    pr_url = None
    if resolution.opens_pr:
        assert resolution.pr_title is not None
        git.push(branch)
        pr_url = gh.create_pr(
            base=settings.base_branch,
            head=branch,
            title=resolution.pr_title,
            body=pr_body(candidate, source, resolution, labels),
        )
        r = Round(r.issue, r.number, r.base, _pr_number(pr_url))
    gh.comment(number, resolution_comment(r, resolution, labels, pr_url=pr_url))
    reason = resolution.close_reason
    if reason is not None:
        gh.close(number, reason.value)
    report.note(
        f"#{number}: {counts}"
        + (f"; PR {pr_url}" if pr_url else "")
        + (f"; labelled {NEEDS_HUMAN_LABEL}" if resolution.needs_human else "")
        + (f"; closed ({reason.value})" if reason else "")
    )


# --------------------------------------------------------------------------- main


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
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
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="main checkout")
    p.add_argument("--remote", default="origin")
    p.add_argument("--base-branch", default="main")
    p.add_argument("--issue", type=int, action="append", default=[],
                   help="only these follow-up issues (repeatable)")  # fmt: skip
    p.add_argument("--dry-run", action="store_true",
                   help="run the agent but write nothing to GitHub and push nothing")  # fmt: skip
    p.add_argument(
        "--agent-timeout", type=float, default=3 * 3600, help="seconds per issue"
    )
    p.add_argument(
        "--agent-exec",
        type=Path,
        default=Path(
            os.environ.get("FOLLOWUP_AGENT_EXEC", HERE.parent / "lib/agent_exec.sh")
        ),
    )
    p.add_argument(
        "--state-dir", type=Path, default=state_home / "tennis-lab-agents/followup"
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    settings = Settings(
        repo=args.repo.resolve(),
        remote=args.remote,
        base_branch=args.base_branch,
        agent_exec=args.agent_exec.resolve(),
        agent_timeout=args.agent_timeout,
        state_dir=args.state_dir,
        dry_run=args.dry_run,
        only=frozenset(args.issue),
    )
    if not os.access(settings.agent_exec, os.X_OK):
        log.error("agent wrapper is not executable: %s", settings.agent_exec)
        return 2
    with exclusive_lock(settings.state_dir / "run.lock") as locked:
        if not locked:
            log.info("another run holds the lock; exiting")
            return EXIT_LOCKED
        scratch = settings.state_dir / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        gh = GitHub(settings.repo, scratch)
        git = Git(settings.repo, settings.remote)
        login = gh.login()
        missing = {NEEDS_HUMAN_LABEL, FOLLOWUP_LABEL} - gh.label_names()
        if missing:
            log.error("missing labels: %s", sorted(missing))
            return 2
        candidates = gh.open_followups(login)
        if settings.only:
            unknown = settings.only - {c.issue.number for c in candidates}
            if unknown:
                log.error("not open follow-up issues by %s: %s", login, sorted(unknown))
                return 2
            candidates = [c for c in candidates if c.issue.number in settings.only]
        report = RunReport()
        report.note(f"{len(candidates)} open follow-up issue(s)")
        for candidate in candidates:
            try:
                resolve_issue(settings, gh, git, login, candidate, report)
            except (CommandError, ResolutionParseError, ValueError, OSError) as exc:
                report.error(f"#{candidate.issue.number}: {exc}")
        print("\n".join(report.lines))
        return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
