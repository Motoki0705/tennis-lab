"""Typed, publisher-validated history across legacy comments and an evidence issue."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .judgement_comments import PREFIX as JUDGEMENT_PREFIX
from .judgement_comments import owned_comment, read_json
from .policy import digest, events, number, object_value

if TYPE_CHECKING:
    from .github import GitHub

Source = Literal["legacy_parent_comment", "evidence_issue"]
Family = Literal["parent-session", "parent-review", "issue-start-judgement"]


class HistoryValidationError(ValueError):
    """Fetched history is invalid; distinguish this from retrieval failures."""


@dataclass(frozen=True)
class Record:
    family: Family
    payload: dict[str, Any]
    source: Source
    comment_id: int
    url: str
    publisher: str
    publisher_verified: bool
    body: str


@dataclass(frozen=True)
class Diagnostic:
    code: str
    source: Source
    comment_id: int
    url: str
    publisher: str
    publisher_verified: bool


@dataclass(frozen=True)
class History:
    records: tuple[Record, ...]
    diagnostics: tuple[Diagnostic, ...]
    parent_comments: list[dict[str, Any]]
    _comments: list[dict[str, Any]]

    def require_valid(self) -> None:
        if self.diagnostics:
            raise HistoryValidationError(
                "; ".join(
                    f"{d.code}: {d.url} publisher={d.publisher}"
                    for d in self.diagnostics
                )
            )

    @property
    def comments(self) -> list[dict[str, Any]]:
        self.require_valid()
        return self._comments

    def session_events(self) -> list[dict[str, Any]]:
        self.require_valid()
        return [
            {**r.payload, "body": r.body}
            for r in self.records
            if r.family == "parent-session"
        ]


def comment_url(repo: str, parent: int, comment: Mapping[str, Any]) -> str:
    issue = number(comment.get("_parent_record_issue", parent))
    return f"https://github.com/{repo}/issues/{issue}#issuecomment-{number(comment.get('id'))}"


def _payload(
    body: str,
    comment: Mapping[str, Any],
    repo: str,
    parent: int,
    login: str,
    *,
    parent_scope: bool = True,
) -> tuple[Family, dict[str, Any]] | None:
    line = body.split("\n", 1)[0].rstrip("\r")
    if line.startswith("<!-- parent-session:v"):
        parsed = events([comment], login, parent)[0]
        return "parent-session", {k: v for k, v in parsed.items() if k != "body"}
    if line.startswith("<!-- parent-review:v"):
        from review.parent_review_policy import records

        parsed = records([comment], login)[0]
        target = parsed.get("parent_issue", parsed.get("parent"))
        if target is None and isinstance(parsed.get("request"), dict):
            target = parsed["request"].get("parent_issue")
        if parent_scope and (
            (target != parent and not (target is None and parsed["kind"] == "pending"))
            or parsed["kind"] == "pr"
        ):
            raise ValueError(
                "parent-review record belongs to another parent or to a PR"
            )
        return "parent-review", parsed
    if line.startswith("<!-- issue-start-judgement:v"):
        if not line.startswith(JUDGEMENT_PREFIX) or not owned_comment(
            comment, login, repo, parent
        ):
            raise ValueError("unknown or malformed judgement marker")
        return "issue-start-judgement", object_value(
            read_json(line[len(JUDGEMENT_PREFIX) : -4])
        )
    return None


def _logical_key(family: Family, payload: dict[str, Any]) -> str:
    """Use recorded revision/run/operation IDs; never fill missing legacy fields."""
    event = payload.get("event")
    fields: tuple[str, ...] = ()
    if family == "parent-session":
        if event in {"checking", "agreed", "execution_basis", "basis_revoked"}:
            fields = ("revision",)
        elif event == "proposal_agreed":
            fields = ("revision", "source")
        elif event in {"proposal_pending", "proposal_received"}:
            fields = ("source",)
        elif event == "agreement_comments":
            fields = ("agreement_sha256", "revision", "classification_revision")
        elif event == "review":
            fields = ("review_run_id", "state")
        elif event == "assigned":
            fields = ("generation",)
        elif event == "note":
            fields = ("generation", "body_sha256")
        elif event in {"child_stopped", "child_resumed"}:
            fields = ("child", "operation")
    elif family == "parent-review":
        event = payload["kind"]
        fields = ("review_run_id",)
        if event in {"resolution", "evidence"}:
            fields += (
                "pr",
                "finding" if event == "resolution" else "wait",
                "fixed_snapshot" if event == "resolution" else "snapshot",
            )
    else:
        fields = ("issue",)
    identity = (
        {key: payload[key] for key in fields}
        if fields and all(key in payload for key in fields)
        else payload
    )
    return digest([family, event, identity])


def _parse_history(
    repo: str,
    parent: int,
    login: str,
    parent_comments: Sequence[Mapping[str, Any]],
    evidence_comments: Sequence[Mapping[str, Any]],
    *,
    evidence_issue: int | None,
) -> History:
    if evidence_issue is None and evidence_comments:
        raise ValueError("evidence comments require a verified issue binding")
    records, diagnostics, comments = [], [], []
    keys: dict[str, Record] = {}
    ids: set[int] = set()
    groups: tuple[tuple[Source, int | None, Sequence[Mapping[str, Any]]], ...] = (
        ("legacy_parent_comment", parent, parent_comments),
        ("evidence_issue", evidence_issue, evidence_comments),
    )
    for source, issue, rows in groups:
        for raw in rows:
            cid = number(raw.get("id"))
            if cid in ids:
                raise ValueError("duplicate comment ID in parent evidence history")
            ids.add(cid)
            body = raw.get("body") or ""
            if not isinstance(body, str):
                raise ValueError("comment body must be text")
            publisher = object_value(raw.get("user")).get("login")
            if not isinstance(publisher, str) or not publisher:
                raise ValueError("comment publisher missing")
            verified = publisher == login
            url = f"https://github.com/{repo}/issues/{issue}#issuecomment-{cid}"
            payload = _payload(body, raw, repo, parent, login) if verified else None
            if source == "evidence_issue" and (not verified or payload is None):
                diagnostics.append(
                    Diagnostic(
                        "untrusted_evidence_publisher"
                        if not verified
                        else "unmarked_evidence_comment",
                        source,
                        cid,
                        url,
                        publisher,
                        verified,
                    )
                )
                continue
            comments.append(
                {**raw, "_parent_record_issue": issue, "_parent_record_source": source}
            )
            if payload is None:
                continue
            family, data = payload
            record = Record(family, data, source, cid, url, publisher, verified, body)
            key = _logical_key(family, data)
            previous = keys.get(key)
            if previous is not None:
                if previous.payload != data:
                    raise ValueError(
                        f"conflicting parent evidence record: {previous.url} / {url}"
                    )
                if previous.source != source:
                    raise ValueError(
                        f"duplicate parent evidence record across media: {previous.url} / {url}"
                    )
            keys[key] = record
            records.append(record)
    return History(
        tuple(sorted(records, key=lambda r: r.comment_id)),
        tuple(diagnostics),
        [dict(c) for c in parent_comments],
        sorted(comments, key=lambda c: c["id"]),
    )


def parse_history(
    repo: str,
    parent: int,
    login: str,
    parent_comments: Sequence[Mapping[str, Any]],
    evidence_comments: Sequence[Mapping[str, Any]],
    *,
    evidence_issue: int | None,
) -> History:
    try:
        return _parse_history(
            repo,
            parent,
            login,
            parent_comments,
            evidence_comments,
            evidence_issue=evidence_issue,
        )
    except ValueError as exc:
        raise HistoryValidationError(str(exc)) from exc


def read_history(
    gh: GitHub,
    parent: int,
    login: str,
    *,
    parent_comments: Sequence[Mapping[str, Any]] | None = None,
) -> History:
    evidence = gh.evidence_issues().find(parent, login)
    return parse_history(
        gh.repo,
        parent,
        login,
        gh.comments(parent) if parent_comments is None else parent_comments,
        gh.comments(evidence.number) if evidence else [],
        evidence_issue=evidence.number if evidence else None,
    )


def from_projection(
    repo: str, parent: int, login: str, comments: Sequence[Mapping[str, Any]]
) -> History:
    """Revalidate a pinned collector projection without network I/O or new fields.

    Live collection verifies the binding on GitHub before making this projection;
    dispatcher freshness recollects it. Old snapshots remain legacy-only.
    """
    legacy, evidence = [], []
    issue = None
    for comment in comments:
        metadata = {"_parent_record_source", "_parent_record_issue"} & comment.keys()
        if metadata and (
            len(metadata) != 2
            or comment["_parent_record_source"]
            not in {"legacy_parent_comment", "evidence_issue"}
            or (
                comment["_parent_record_source"] == "legacy_parent_comment"
                and comment["_parent_record_issue"] != parent
            )
        ):
            raise HistoryValidationError("invalid pinned parent evidence provenance")
        if comment.get("_parent_record_source") == "evidence_issue":
            n = number(comment.get("_parent_record_issue"))
            if issue is not None and issue != n:
                raise ValueError("multiple evidence issues in a pinned history")
            issue = n
            evidence.append(comment)
        else:
            legacy.append(comment)
    return parse_history(repo, parent, login, legacy, evidence, evidence_issue=issue)


def instruction_comments(
    repo: str, parent: int, login: str, comments: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    from .execution import comment_snapshot

    result = []
    for comment in comments:
        item = comment_snapshot(comment)
        if (
            item["login"] == login
            and _payload(item["body"], comment, repo, parent, login, parent_scope=False)
            is not None
        ):
            continue
        result.append(item)
    return result
