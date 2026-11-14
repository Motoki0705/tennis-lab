"""Strict, versioned facts added to v2 results; v1 records remain immutable."""

from __future__ import annotations

import re
from typing import Any

from worker_logs.journal import require
from worker_logs.schema import digest, fields, integer

from .schema import choice, evidence_list, text, timestamp


def proof(value: Any) -> None:
    evidence_list(value)
    require(bool(value), "record_source")


def nullable_text(value: Any) -> None:
    if value is not None:
        text(value)


def revision(value: Any) -> None:
    require(value is None or digest(value, 40), "record_source")


def fact(value: Any, states: tuple[str, ...]) -> None:
    fields(value, {"state", "evidence"})
    choice(value["state"], states)
    proof(value["evidence"])


def remaining(value: Any, parent: int, *, version: int) -> None:
    require(isinstance(value, list), "record_schema")
    identities = set()
    common = {"kind", "parent", "state", "stage", "error", "pr_number", "pr_head"}
    for row in value:
        require(isinstance(row, dict), "record_schema")
        choice(row.get("kind"), ("worker_log", "decision_log"))
        fields(
            row,
            common
            | (
                {"issue", "public_key", "sequence", "unpublished_count", "batch_key"}
                if version == 2
                else {"issue", "public_key", "sequence"}
                if row["kind"] == "worker_log"
                else {"unpublished_count", "batch_key"}
            ),
        )
        require(integer(row["parent"]) and row["parent"] == parent, "record_issue")
        text(row["state"])
        for name in ("stage", "error"):
            nullable_text(row[name])
        require(row["pr_number"] is None or integer(row["pr_number"]), "record_source")
        revision(row["pr_head"])
        identity: tuple[Any, ...]
        if row["kind"] == "worker_log":
            require(
                all(
                    integer(row[k]) or version == 2 and row[k] is None
                    for k in ("issue", "sequence")
                ),
                "record_turn",
            )
            require(
                version == 2
                and row["public_key"] is None
                or isinstance(row["public_key"], str)
                and bool(re.fullmatch(r"d-[a-f0-9]{32}", row["public_key"])),
                "record_key",
            )
            if version == 2:
                require(
                    row["unpublished_count"] is None and row["batch_key"] is None,
                    "record_schema",
                )
            identity = (row["kind"], row["issue"], row["public_key"], row["sequence"])
        else:
            require(
                version == 2
                and row["unpublished_count"] is None
                or type(row["unpublished_count"]) is int
                and row["unpublished_count"] >= 0,
                "record_schema",
            )
            require(
                row["batch_key"] is None or digest(row["batch_key"], 32), "record_key"
            )
            if version == 2:
                require(
                    row["issue"] is None
                    and row["sequence"] is None
                    and row["public_key"] == row["batch_key"],
                    "record_source",
                )
            identity = (row["kind"], row["batch_key"])
        require(identity not in identities, "record_already_exists")
        identities.add(identity)


def validate(value: Any, scope: dict[str, Any], final_report: dict[str, Any]) -> None:
    require(isinstance(value, dict), "record_schema")
    version = value.get("version")
    require(type(version) is int and version in {1, 2}, "record_version")
    choice(value.get("kind"), ("worker_turn", "termination", "capture_gap"))
    require(version == 1 or value["kind"] == "termination", "record_version")
    if value["kind"] == "worker_turn":
        fields(value, {"version", "kind", "execution", "final", "publication"})
        require(scope["delegation"] is not None, "record_turn")
        fact(value["execution"], ("running", "ended", "not_started", "unknown"))
        fact(value["final"], ("present", "absent", "unknown"))
        require(
            value["final"]["state"] != "absent"
            or value["execution"]["state"] == "ended",
            "record_schema",
        )
        require(
            (value["final"]["state"] == "present")
            == (final_report["status"] == "observed"),
            "record_schema",
        )
        if value["final"]["state"] == "absent":
            require(
                final_report == {"status": "unavailable", "reason": "missing_final"},
                "record_schema",
            )
        public = value["publication"]
        fields(
            public,
            {
                "state",
                "stage",
                "error",
                "pr_url",
                "pr_head",
                "main_revision",
                "evidence",
            },
        )
        choice(public["state"], ("main", "public_pr", "pending", "unknown"))
        proof(public["evidence"])
        for name in ("stage", "error"):
            nullable_text(public[name])
        for name in ("pr_head", "main_revision"):
            revision(public[name])
        if public["pr_url"] is not None:
            require(
                isinstance(public["pr_url"], str)
                and bool(
                    re.fullmatch(
                        r"https://github\.com/"
                        + re.escape(scope["repo"])
                        + r"/pull/[1-9][0-9]*",
                        public["pr_url"],
                        re.IGNORECASE,
                    )
                ),
                "record_source",
            )
        if public["state"] == "public_pr":
            require(
                public["pr_url"] is not None and public["pr_head"] is not None,
                "record_source",
            )
        if public["state"] == "main":
            require(public["main_revision"] is not None, "record_source")
    elif value["kind"] == "termination":
        fields(
            value,
            {
                "version",
                "kind",
                "generation",
                "event",
                "observed_at",
                "main_revision",
                "acceptance",
                "evidence",
                "remaining",
                "publication_actor",
                "resume_command",
            }
            | (
                {"remaining_complete", "error", "decision_resume_command"}
                if version == 2
                else {"remaining_status"}
            ),
        )
        require(scope["delegation"] is None and scope["issue"] is None, "record_turn")
        require(integer(value["generation"]), "record_schema")
        choice(value["event"], ("ended", "handoff_pending", "handoff_accepted"))
        timestamp(value["observed_at"])
        revision(value["main_revision"])
        fact(value["acceptance"], ("completed", "incomplete", "unverified"))
        proof(value["evidence"])
        remaining(value["remaining"], scope["parent"], version=version)
        if version == 2:
            require(type(value["remaining_complete"]) is bool, "record_schema")
            require(
                integer(value["publication_actor"])
                or isinstance(value["publication_actor"], str),
                "record_schema",
            )
            if isinstance(value["publication_actor"], str):
                text(value["publication_actor"])
            nullable_text(value["error"])
            nullable_text(value["decision_resume_command"])
        else:
            choice(value["remaining_status"], ("observed", "unknown"))
            require(
                value["remaining_status"] != "unknown" or not value["remaining"],
                "record_schema",
            )
            nullable_text(value["publication_actor"])
        nullable_text(value["resume_command"])
    else:
        fields(value, {"version", "kind", "status", "reason", "evidence"})
        require(value["status"] == "needs_reconciliation", "record_schema")
        text(value["reason"])
        proof(value["evidence"])
        require(len(value["evidence"]) >= 2, "record_source")
