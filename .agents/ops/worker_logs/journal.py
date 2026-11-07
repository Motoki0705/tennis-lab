"""Private per-turn storage. Publication is never inferred from a process exit."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PUBLIC_ERRORS = frozenset(
    [
        "absolute_path",
        "ambiguous_encoding",
        "closed_unmerged",
        "committed_content_changed",
        "control_character",
        "credential",
        "duplicate_json_key",
        "empty_text",
        "empty_worker_instruction",
        "final_not_ready",
        "git_failed",
        "github_failed",
        "github_invalid_response",
        "github_pagination",
        "input_capture_failed",
        "instruction_unavailable",
        "invalid_error_code",
        "invalid_json_object",
        "invalid_log_reference",
        "invalid_public_key",
        "invalid_repository_layout",
        "journal_identity_changed",
        "journal_invalid",
        "journal_key",
        "journal_pr",
        "journal_source",
        "journal_stage",
        "journal_state",
        "journal_turn",
        "journal_unavailable",
        "journal_version",
        "log_operation_failed",
        "log_owner_mismatch",
        "log_owner_unavailable",
        "main_content_changed",
        "merge_conflict",
        "merge_response_uncertain",
        "merge_unconfirmed",
        "missing_final",
        "multiple_log_prs",
        "no_captured_turn",
        "pr_creation_uncertain",
        "pr_identity_changed",
        "prepared_content_changed",
        "private_identifier",
        "private_storage_symlink",
        "private_value",
        "public_key_missing",
        "publication_failed",
        "publication_identity_changed",
        "receipt_unavailable",
        "receipt_unverifiable",
        "received_final_changed",
        "record_already_exists",
        "record_artifact",
        "record_decision",
        "record_digest",
        "record_files",
        "record_instruction",
        "record_issue",
        "record_key",
        "record_phase",
        "record_repository",
        "record_runtime",
        "record_sanitization",
        "record_schema",
        "record_source",
        "record_time",
        "record_turn",
        "record_version",
        "record_worker",
        "reference_unavailable",
        "remote_head_changed",
        "repository_changed",
        "repository_unavailable",
        "runner_contract_changed",
        "runner_evidence_unavailable",
        "runner_not_ended",
        "runner_settings_changed",
        "runner_time_unavailable",
        "runtime_changed",
        "runtime_reference_unavailable",
        "sent_prompt_changed",
        "source_changed",
        "standard_unverifiable",
        "unexpected_commit_parent",
        "unsafe_diff",
        "unsafe_file_type",
        "unsafe_reference",
        "unsafe_url",
        "worker_failed",
        "worker_head_unavailable",
        "worktree_changed",
        "worktree_dirty",
        "worktree_symlink",
    ]
)


class PublicationError(ValueError):
    """Only a stable public code may cross the publisher boundary."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        super().__init__(code if code in PUBLIC_ERRORS else "publication_failed")
        self.detail = detail if code in PUBLIC_ERRORS else detail or code


def require(condition: bool, code: str) -> None:
    if not condition:
        raise PublicationError(code)


def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    require(len(pairs) == len({k for k, _ in pairs}), "duplicate_json_key")
    return dict(pairs)


def decode(raw: bytes | str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        require(isinstance(value, dict), "invalid_json_object")
        return dict(value)
    except (ValueError, UnicodeError, TypeError):
        raise PublicationError("invalid_json_object") from None


def private_directory(path: Path) -> None:
    missing = []
    ancestor = path
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    require(not ancestor.is_symlink(), "private_storage_symlink")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        require(not directory.is_symlink(), "private_storage_symlink")


def atomic(path: Path, value: dict[str, Any]) -> None:
    private_directory(path.parent)
    require(not path.is_symlink(), "private_storage_symlink")
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".journal-")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(path: Path) -> Iterator[None]:
    private_directory(path.parent)
    fd = os.open(
        path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def read(path: Path) -> dict[str, Any]:
    try:
        require(not path.is_symlink(), "private_storage_symlink")
        row = decode(path.read_bytes())
        require(
            type(row.get("version")) is int and row["version"] == 1, "journal_version"
        )
        require(isinstance(row.get("source"), dict), "journal_source")
        require(
            row.get("state")
            in {
                "captured",
                "ready",
                "pending",
                "merged",
                "blocked",
                "worker_failed",
                "missing_final",
                "legacy",
            },
            "journal_state",
        )
        if row["state"] in {"ready", "pending", "merged"}:
            require(
                isinstance(row.get("files"), dict)
                and set(row["files"]) == {"instruction.md", "final.md", "meta.json"}
                and isinstance(row.get("receipt"), dict),
                "journal_source",
            )
        if row["state"] == "merged":
            require(
                row.get("stage") == "merged"
                and type(row.get("pr_number")) is int
                and row["pr_number"] > 0,
                "journal_pr",
            )
            require(
                all(
                    isinstance(row.get(k), str)
                    and re.fullmatch(r"[a-f0-9]{40}", row[k])
                    for k in ("base", "head", "merge_commit", "main_observed")
                ),
                "journal_source",
            )
        return row
    except (OSError, ValueError, KeyError, TypeError):
        raise PublicationError("journal_invalid") from None


def fail(path: Path, row: dict[str, Any], code: str, *, state: str = "blocked") -> None:
    require(code in PUBLIC_ERRORS, "invalid_error_code")
    row.update(state=state, error=code)
    atomic(path, row)


def view(row: dict[str, Any]) -> dict[str, Any]:
    source = row["source"]
    require(
        type(source.get("sequence")) is int and source["sequence"] > 0, "journal_turn"
    )
    key = source.get("public_key")
    require(
        isinstance(key, str) and bool(re.fullmatch(r"[a-f0-9]{32}", key)), "journal_key"
    )
    code = row.get("error")
    require(code is None or code in PUBLIC_ERRORS, "invalid_error_code")
    number = row.get("pr_number")
    require(number is None or type(number) is int and number > 0, "journal_pr")
    require(
        row.get("stage")
        in {
            None,
            "awaiting_final",
            "receipt_verified",
            "committed",
            "pushed",
            "pr_create",
            "pr_open",
            "merge_attempted",
            "merged",
        },
        "journal_stage",
    )
    return {
        "delegation_key": "d-" + key,
        "sequence": source["sequence"],
        "state": row["state"],
        "stage": row.get("stage"),
        "error": code,
        "pr": number,
        "saved_to_main": row["state"] == "merged",
    }
