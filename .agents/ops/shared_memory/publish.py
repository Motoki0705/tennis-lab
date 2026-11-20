"""Publish one checked shared-memory batch on the latest GitHub main."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

from .git import Git, GitTree, diff_changes
from .github import GitHub, Pull, RefUpdateRejected
from .operations import (
    Operation,
    parse_operations,
    plan_operations,
    publication_reason,
    validation_error,
)
from .validate import validate

MAX_RETRIES = 3


@dataclass(frozen=True)
class OperationSummary:
    op: str
    layer: str
    slug: str
    reason: str


@dataclass(frozen=True)
class Publication:
    dry_run: bool
    base: str
    head: str
    files: dict[str, str]
    operations: tuple[OperationSummary, ...]
    validation: str = "passed"
    pr: int | None = None
    url: str | None = None
    merge_sha: str | None = None
    retries: int = 0
    branch_cleanup_error: str | None = None
    index_repair_reason: str | None = None


@dataclass(frozen=True)
class _Prepared:
    base: str
    head: str
    files: dict[str, str]


class _Retry(ValueError):
    """Retry only after observing a forward move of main in Git."""


def _prepare(
    git: Git,
    remote: str,
    operations: Sequence[Operation],
    today: date,
    previous_base: str | None,
    retry_reason: str,
    *,
    require_fresh: bool,
) -> _Prepared:
    base = git.fetch_main(remote)
    if previous_base is not None:
        if base == previous_base:
            raise ValueError(f"{retry_reason}; remote main has not advanced")
        if not git.is_ancestor(previous_base, base):
            raise ValueError("remote main was rewritten during publication")
    before = GitTree(git, base)
    edits = plan_operations(before, operations, today=today)
    if not edits:
        raise ValueError("generated documents are already up to date (no-op)")
    expected = {
        path: "removed"
        if content is None
        else ("modified" if before.kind(path, follow_links=False) else "added")
        for path, content in edits.items()
    }
    head = git.build_commit(base, edits)
    if git.run("rev-list", "--parents", "-n", "1", head).decode().split() != [
        head,
        base,
    ]:
        raise ValueError("memory commit must have the fetched main as its only parent")
    diff = ("diff", "--raw", "--abbrev=40", "--no-ext-diff", "--ignore-submodules=none")
    raw = git.run(*diff, "--no-renames", "-z", base, head, "--")
    changes = diff_changes(raw)
    if changes != expected:
        raise ValueError("committed diff differs from the requested operations")
    # --no-renames establishes both paths and modes. Also reject a detected rename.
    if git.run(*diff, "--find-renames", "-l0", "-z", base, head, "--") != raw:
        raise ValueError("rename detected in committed diff")
    after = GitTree(git, head)
    for path, content in edits.items():
        if content is not None and after.read_bytes(path) != content:
            raise ValueError(f"{path}: committed bytes differ from the validated edit")
    if problems := validate(after, today=today, require_fresh=require_fresh):
        raise validation_error(problems)
    return _Prepared(base, head, changes)


def _body(
    operations: Sequence[Operation], files: dict[str, str], repair_reason: str | None
) -> str:
    labels = {"added": "追加", "modified": "更新", "removed": "削除"}
    actions = {"add": "追加", "update": "更新", "delete": "削除"}
    lines = ["## 変更ファイル", ""]
    lines.extend(
        f"- {labels[status]} `{path}`" for path, status in sorted(files.items())
    )
    lines.extend(("", "## 操作の理由"))
    for operation in operations:
        lines.extend(("", f"**{actions[operation.op]} `{operation.path}`**", ""))
        lines.extend(f"    {line}" for line in operation.reason.splitlines())
    if repair_reason is not None:
        lines.extend(("", "**生成文書を再生成**", ""))
        lines.extend(f"    {line}" for line in repair_reason.splitlines())
    lines.extend(
        (
            "",
            "## 機械検査",
            "",
            "共有4層の形式・参照・上限・公開内容と、生成文書のバイト一致を確認しました。",
            "main を親とする1コミットの変更パス・モード・内容を照合しました。",
        )
    )
    return "\n".join(lines)


def _advance_checked(
    github: GitHub, number: int, branch: str, body: str, candidate: _Prepared
) -> None:
    def verify_state() -> Pull:
        pull = github.pull(number, branch, candidate.head, body)
        if github.main() != candidate.base:
            raise _Retry("GitHub reports that main changed before merge")
        if pull.changed_files != len(candidate.files):
            raise ValueError(
                "GitHub pull request file count differs from committed diff"
            )
        return pull

    verify_state()
    github.verify_files(number, candidate.files)
    pull = verify_state()
    try:
        github.advance_main(pull.repository_id, branch, candidate.base, candidate.head)
    except RefUpdateRejected as exc:
        raise _Retry(str(exc)) from None


def publish_operations(
    repo: Path,
    operations: Sequence[Mapping[str, object]],
    *,
    remote: str = "origin",
    dry_run: bool = False,
    today: date | None = None,
    require_fresh: bool = False,
) -> Publication:
    """Validate, publish and merge a batch, raising on refused or uncertain steps."""
    today = today or date.today()
    return _publish(
        repo,
        parse_operations(operations, today),
        today,
        remote,
        dry_run,
        require_fresh=require_fresh,
    )


def publish_index_repair(
    repo: Path,
    *,
    reason: str,
    remote: str = "origin",
    dry_run: bool = False,
    today: date | None = None,
    require_fresh: bool = False,
) -> Publication:
    """Publish only regenerated documents when the latest main has a repair diff."""
    return _publish(
        repo,
        (),
        today or date.today(),
        remote,
        dry_run,
        repair_reason=publication_reason(reason),
        require_fresh=require_fresh,
    )


def _publish(
    repo: Path,
    checked: Sequence[Operation],
    today: date,
    remote: str,
    dry_run: bool,
    *,
    require_fresh: bool,
    repair_reason: str | None = None,
) -> Publication:
    summaries = tuple(
        OperationSummary(o.op, o.layer, o.slug, o.reason) for o in checked
    )
    git = Git(repo)
    github = GitHub(git.github_repo(remote))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    branch = f"shared-memory/{timestamp}-{uuid4().hex[:12]}"
    previous: _Prepared | None = None
    retry_reason = ""
    number = None
    push_attempted = False
    old_body = ""
    retries = 0

    def verify_remote() -> None:
        if git.github_repo(remote).casefold() != github.repo.casefold():
            raise ValueError("configured remote changed during publication")

    try:
        while True:
            verify_remote()
            candidate = _prepare(
                git,
                remote,
                checked,
                today,
                previous.base if previous else None,
                retry_reason,
                require_fresh=require_fresh,
            )
            result = Publication(
                dry_run,
                candidate.base,
                candidate.head,
                candidate.files,
                summaries,
                index_repair_reason=repair_reason,
            )
            if dry_run:
                return result
            if previous is not None and number is not None:
                github.pull(number, branch, previous.head, old_body)
            verify_remote()
            push_attempted = True
            git.push(
                remote,
                branch,
                candidate.head,
                expected=previous.head if previous else "",
            )
            body = _body(checked, candidate.files, repair_reason)
            if number is None:
                number = github.create(branch, body)
            elif body != old_body:
                github.update_body(number, body)
            try:
                _advance_checked(github, number, branch, body, candidate)
            except _Retry as exc:
                if retries == MAX_RETRIES:
                    raise ValueError(
                        f"{exc}; retry limit reached ({MAX_RETRIES})"
                    ) from None
                previous, old_body, retry_reason = candidate, body, str(exc)
                retries += 1
            else:
                break
        try:
            verify_remote()
            if git.remote_main(remote) != candidate.head and not git.is_ancestor(
                candidate.head, git.fetch_main(remote)
            ):
                raise ValueError("the published head is not reachable from remote main")
            pull = github.pull(number, branch, candidate.head, body, merged=True)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"main update acknowledged but merge confirmation failed: {exc}"
            ) from None
    except (OSError, ValueError) as exc:
        if not push_attempted:
            raise
        identity = f"PR #{number}, {branch}" if number is not None else branch
        raise ValueError(f"{identity}: {exc}") from None

    cleanup_error = None
    try:
        verify_remote()
        git.push(remote, branch, "", expected=candidate.head)
    except (OSError, ValueError) as exc:
        cleanup_error = f"{branch}: {exc}"
    return replace(
        result,
        pr=number,
        url=pull.url,
        merge_sha=candidate.head,
        retries=retries,
        branch_cleanup_error=cleanup_error,
    )
