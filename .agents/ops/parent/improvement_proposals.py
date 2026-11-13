"""Proposal projection, explicit grouped extraction and daily expiry.

Callers hold improvements.lock_key(repo). Business events remain source comments;
receipts only recover uncertain writes and never grant execution permission.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from issues.cli import render_spec
from issues.creation import create_issue_once
from issues.format import CHILD_LABEL, PARENT_LABEL, PROPOSAL_LABEL, STANDALONE_LABEL
from issues.improvement import (
    ImprovementSource,
    candidate_key,
    contract,
    event,
    fields,
    fingerprint,
    human_state,
    key,
    positive,
)
from issues.improvement_proposals import (
    Extraction,
    Proposal,
    parse_extraction,
    parse_proposal,
)
from issues.supplement_state import _utc_timestamp
from worker_logs.journal import atomic, decode

from .improvements import append, candidates, comment_url

if TYPE_CHECKING:
    from .session import ParentSession


def receipt_path(session: ParentSession, name: str) -> Path:
    return (
        Path(session.store.directory)
        / "improvement-proposals"
        / fingerprint(session.gh.repo.casefold())
        / f"{name}.json"
    )


def load_receipt(session: ParentSession, name: str) -> dict[str, Any] | None:
    path = receipt_path(session, name)
    if path.is_symlink():
        raise ValueError("improvement receipt must not be a symlink")
    if not path.exists():
        return None
    value = decode(path.read_bytes())
    if not isinstance(value, dict) or value.get("repo") != session.gh.repo.casefold():
        raise ValueError("improvement receipt identity mismatch")
    positive(value.get("parent"))
    creation = fields(value.get("creation"), {"draft", "attempted", "number"})
    if type(creation["attempted"]) is not bool:
        raise ValueError("invalid improvement creation attempt")
    if creation["number"] is not None:
        positive(creation["number"])
    if creation["draft"] is not None:
        fields(creation["draft"], {"title", "body", "labels"})
    if name.startswith("proposal-"):
        fields(value, {"repo", "parent", "generation", "members", "creation"})
        source = Proposal(value["parent"], value["generation"])
        if (
            name != f"proposal-{source.parent}"
            or not isinstance(value["members"], dict)
            or not value["members"]
        ):
            raise ValueError("invalid proposal receipt membership")
        for k, revision in value["members"].items():
            key(k)
            positive(revision)
        if (
            creation["draft"] is not None
            and parse_proposal(creation["draft"]["body"]) != source
        ):
            raise ValueError("proposal receipt provenance mismatch")
    else:
        fields(
            value, {"repo", "parent", "keys", "proposal", "decision_ids", "creation"}
        )
        extracted_source = Extraction(
            value["parent"], value["proposal"], tuple(value["keys"])
        )
        if name != f"extraction-{extracted_source.proposal}" or set(
            value["decision_ids"]
        ) != set(extracted_source.keys):
            raise ValueError("invalid extraction receipt approvals")
        if (
            creation["draft"] is None
            or parse_extraction(creation["draft"]["body"]) != extracted_source
        ):
            raise ValueError("extraction receipt provenance mismatch")
    return dict(value)


def save_receipt(session: ParentSession, name: str, receipt: dict[str, Any]) -> None:
    atomic(receipt_path(session, name), receipt)


def index_proposals(
    session: ParentSession, rows: list[dict[str, Any]], recorded: dict[str, Any]
) -> dict[Proposal, dict[str, Any]]:
    index: dict[Proposal, dict[str, Any]] = {}
    for raw in rows:
        source = parse_proposal(raw.get("body") or "")
        if source is None or source.parent != session.parent:
            continue
        if source in index:
            raise ValueError("duplicate improvement proposal generation")
        if raw["state"] == "open" and raw.get("state_reason") == "reopened":
            raise ValueError(
                "closed proposal must not be reopened; close it before publishing new candidates"
            )
        index[source] = raw
    for row in recorded.values():
        membership = row["proposal"]
        if membership is None:
            continue
        data = membership["data"]
        source = Proposal(session.parent, data["generation"])
        found = index.get(source)
        if (
            found is None
            or found["number"] != data["issue"]
            or found["created_at"] != data["created_at"]
        ):
            raise ValueError(
                "recorded proposal disappeared or lost its immutable source/created_at; do not recreate"
            )
    opened = [raw for raw in index.values() if raw["state"] == "open"]
    if len(opened) > 1:
        raise ValueError("source parent has more than one open proposal")
    return index


def proposal_issue(
    session: ParentSession, membership: dict[str, Any]
) -> dict[str, Any]:
    data = membership["data"]
    n = positive(data["issue"])
    raw = session.gh.get(f"repos/{session.gh.repo}/issues/{n}")
    if (
        raw["number"] != n
        or parse_proposal(raw.get("body") or "")
        != Proposal(session.parent, data["generation"])
        or raw["created_at"] != data["created_at"]
    ):
        raise ValueError("proposal provenance changed")
    return dict(raw)


def members(recorded: dict[str, Any], issue: int) -> dict[str, Any]:
    return {
        k: row
        for k, row in recorded.items()
        if row["proposal"] is not None and row["proposal"]["data"]["issue"] == issue
    }


def eligible(recorded: dict[str, Any]) -> dict[str, Any]:
    return {
        k: row
        for k, row in recorded.items()
        if row["proposal"] is None
        and row["publication"] is None
        and row["extraction"] is None
    }


def validate_candidate(row: dict[str, Any]) -> None:
    if row["candidate"]["data"]["legacy"]:
        raise ValueError(
            "candidate boundary is incomplete; update the same key before proposing"
        )
    if row["candidate"]["data"]["spec"]["unmet_acceptance"]:
        raise ValueError(
            "unfinished original acceptance must remain with the original task"
        )


def candidate_requirements(spec: dict[str, Any]) -> dict[str, str]:
    """Freeze every candidate field, including exclusions and stopping rules."""
    sections = {
        "目的・背景": spec["purpose"] + "\n\n" + spec["observation"],
        "全体のスコープ": "対象:\n"
        + spec["scope"]
        + "\n\n対象外:\n"
        + spec["out_of_scope"],
        "全体方針・合意事項": spec["behavior"],
        "全体の完了条件": "\n".join("- [ ] " + x for x in spec["acceptance"]),
        "依存関係・進め方": "停止条件:\n" + spec["stop_conditions"],
        "観測・根拠と切り出し判断": spec["evidence"] + "\n\n" + spec["judgment"],
        "未決事項・人間の判断が必要な点": spec["unknowns"],
        "期待する効果": spec["expected_effect"],
    }
    # Same normalization as the existing execution contract, without an Issue body.
    return contract("\n\n".join(f"## {h}\n\n{v}" for h, v in sections.items()))


def deadline(raw: dict[str, Any]) -> datetime:
    return cast(
        datetime, _utc_timestamp(raw["created_at"], "proposal created_at")
    ) + timedelta(days=7)


def render_proposal(
    source: Proposal,
    raw: dict[str, Any] | None,
    recorded: dict[str, Any],
    *,
    ended: str | None = None,
) -> str:
    due = (
        deadline(raw).strftime("%Y-%m-%dT%H:%M:%SZ")
        if raw is not None
        else "作成日時の確定後に表示"
    )
    lines = [
        source.marker(),
        f"元親: #{source.parent} / generation {source.generation}",
        f"期限（UTC、作成から7日）: {due}",
        "判断は会話で親AIへ伝えてください。本文の編集・チェック・ラベル操作は承認になりません。承認分は人間の指示を受けて1件のparentへ取り出します。",
    ]
    if ended:
        lines.append("この提案は終了しています。追加の承認・取り出しはできません。")
    labels = {
        "pending": "未判断",
        "approved": "承認",
        "held": "保留",
        "declined": "見送り",
        "revoked": "撤回",
    }
    for k, row in sorted(recorded.items()):
        spec = row["candidate"]["data"]["spec"]
        extraction = row["extraction"]
        state = (
            f"取り出し済み → #{extraction['data']['issue']}"
            if extraction
            else ended or labels[human_state(row)]
        )
        lines.extend(
            [f"## I-{source.parent}-{k} / revision {row['revision']}", f"状態: {state}"]
        )
        for title, value in [
            ("目的", spec["purpose"]),
            ("観測", spec["observation"]),
            ("根拠", spec["evidence"]),
            ("対象範囲", spec["scope"]),
            ("対象外", spec["out_of_scope"]),
            ("期待する動作", spec["behavior"]),
            ("受入条件", "\n".join("- " + x for x in spec["acceptance"])),
            ("切り出し判断", spec["judgment"]),
            ("停止条件", spec["stop_conditions"]),
            ("未確認事項", spec["unknowns"]),
            ("期待する効果", spec["expected_effect"]),
        ]:
            lines.append(f"### {title}\n\n{value}")
    return "\n\n".join(lines) + "\n"


def refresh(
    session: ParentSession,
    raw: dict[str, Any],
    *,
    ended: str | None = None,
    close: str | None = None,
) -> None:
    source = parse_proposal(raw.get("body") or "")
    if source is None or source.parent != session.parent:
        raise ValueError("proposal source is missing")
    body = render_proposal(
        source, raw, members(candidates(session), raw["number"]), ended=ended
    )
    payload: dict[str, Any] = {}
    if body != raw.get("body"):
        payload["body"] = body
    if close is not None and raw["state"] != "closed":
        payload.update(state="closed", state_reason=close)
    if payload:
        session.public(payload)
        session.gh.write(
            "PATCH", f"repos/{session.gh.repo}/issues/{raw['number']}", payload
        )


def propose(session: ParentSession) -> dict[str, Any]:
    if session.row.get("status") in {"active", "handoff_pending"}:
        session.owner()
    recorded = candidates(session)
    rows = session.gh.issue_rows()  # all states, all pages; never a search index
    index = index_proposals(session, rows, recorded)
    available = eligible(recorded)
    name = f"proposal-{session.parent}"
    receipt = load_receipt(session, name)
    opened = [(s, r) for s, r in index.items() if r["state"] == "open"]
    pending = receipt is not None and any(
        recorded[k]["proposal"] is None for k in receipt["members"]
    )
    if pending:
        assert receipt is not None
        source = Proposal(session.parent, receipt["generation"])
        raw = index.get(source)
    elif opened:
        source, raw = opened[0]
    elif available:
        source = Proposal(
            session.parent,
            max(
                [s.generation for s in index]
                + [receipt["generation"] if receipt else 0]
            )
            + 1,
        )
        raw = None
        receipt = None
    else:
        return {"issue": None, "candidates": [], "unpublished": []}
    for row in available.values():
        validate_candidate(row)
    if raw is None or pending:
        if receipt is None:
            receipt = {
                "repo": session.gh.repo.casefold(),
                "parent": session.parent,
                "generation": source.generation,
                "members": {k: r["revision"] for k, r in sorted(available.items())},
                "creation": {"draft": None, "attempted": False, "number": None},
            }
        if (
            receipt["parent"] != session.parent
            or receipt["generation"] != source.generation
        ):
            raise ValueError("proposal receipt identity mismatch")
        for k, revision in receipt["members"].items():
            if recorded[k]["revision"] != revision:
                raise ValueError(
                    "candidate changed during proposal publication; recover the fixed request"
                )
        draft = receipt["creation"]["draft"] or {
            "title": f"改善提案: 元親 #{session.parent}",
            "body": render_proposal(
                source, None, {k: recorded[k] for k in receipt["members"]}
            ),
            "labels": sorted([PROPOSAL_LABEL, "needs-human", "enhancement"]),
        }
        session.public(draft)
        result = create_issue_once(
            session.gh,
            draft=draft,
            receipt=receipt["creation"],
            matches=[raw] if raw else [],
            save=lambda: save_receipt(session, name, receipt),
        )
        raw = dict(
            session.gh.get(
                f"repos/{session.gh.repo}/issues/{positive(result['number'])}"
            )
        )
        if parse_proposal(raw.get("body") or "") != source:
            raise ValueError("created proposal source mismatch")
        selected = dict(receipt["members"])
        if raw["state"] == "open":
            selected.update({k: r["revision"] for k, r in available.items()})
    else:
        selected = {k: r["revision"] for k, r in available.items()}
    labels = {x["name"] for x in raw["labels"]}
    if (
        PROPOSAL_LABEL not in labels
        or {PARENT_LABEL, CHILD_LABEL, STANDALONE_LABEL} & labels
    ):
        raise ValueError("proposal has invalid role labels")
    n = positive(raw["number"])
    if session.gh.get(
        f"repos/{session.gh.repo}/issues/{n}/parent", missing_parent=True
    ) is not None or session.gh.rows(f"repos/{session.gh.repo}/issues/{n}/sub_issues"):
        raise ValueError("proposal cannot have formal parent/children")
    for k in sorted(selected):
        row = candidates(session)[k]
        if row["proposal"] is None:
            append(
                session,
                event(
                    ImprovementSource(session.parent, k),
                    "proposed",
                    row["revision"],
                    row["last"]["id"],
                    {
                        "issue": n,
                        "generation": source.generation,
                        "created_at": raw["created_at"],
                    },
                ),
            )
    ended = (
        "期限切れ"
        if raw.get("state_reason") == "not_planned"
        else "提案終了"
        if raw["state"] == "closed"
        else None
    )
    refresh(session, raw, ended=ended)
    return {
        "issue": n,
        "generation": source.generation,
        "candidates": [
            f"I-{session.parent}-{k}" for k in sorted(members(candidates(session), n))
        ],
        "reused": bool(opened or pending),
    }


def extraction_draft(
    session: ParentSession, source: Extraction, selected: dict[str, Any]
) -> dict[str, Any]:
    snapshots = {
        k: r["decisions"][-1]["data"]["requirements"] for k, r in selected.items()
    }
    headings = sorted({h for snapshot in snapshots.values() for h in snapshot})
    sections = {
        h: "\n\n".join(
            f"### I-{source.parent}-{k}\n\n{v[h]}"
            for k, v in sorted(snapshots.items())
            if h in v
        )
        for h in headings
    }
    sections["進捗概要"] = "未着手。掲載した候補の人間承認を引き継ぐ。"
    sections["改善の出典"] = (
        f"提案Issue: https://github.com/{session.gh.repo.casefold()}/issues/{source.proposal}\n\n"
        + "\n".join(
            f"- [{ImprovementSource(source.parent, k).candidate_id}]({comment_url(session, source.parent, r['comments'][r['decisions'][-1]['id']])}) / revision {r['revision']}"
            for k, r in sorted(selected.items())
        )
    )
    result = render_spec(
        {
            "role": "parent",
            "identifier": "継続改善",
            "summary": f"元親 #{source.parent} の承認済み改善（{len(selected)}件）",
            "children": [],
            "labels": ["enhancement"],
            "sections": sections,
        }
    )
    return {
        "title": result["title"],
        "body": source.marker() + "\n\n" + str(result["body"]),
        "labels": result["labels"],
    }


def extract(session: ParentSession, requested: list[str]) -> dict[str, Any]:
    keys = sorted(candidate_key(value, session.parent) for value in requested)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("extraction needs unique candidates")
    recorded = candidates(session)
    if any(k not in recorded or recorded[k]["proposal"] is None for k in keys):
        raise ValueError("extraction candidates must belong to a recorded proposal")
    proposals = {recorded[k]["proposal"]["data"]["issue"] for k in keys}
    if len(proposals) != 1:
        raise ValueError("extraction candidates belong to different proposals")
    proposal = proposals.pop()
    source = Extraction(session.parent, proposal, tuple(keys))
    raw = proposal_issue(session, recorded[keys[0]]["proposal"])
    selected = {k: recorded[k] for k in keys}
    name = f"extraction-{proposal}"
    receipt = load_receipt(session, name)
    if receipt is not None and receipt["keys"] != keys:
        raise ValueError(
            "proposal already has a different fixed extraction candidate set"
        )
    recovering = receipt is not None and receipt["creation"]["attempted"]
    extracted = [
        r["extraction"] for r in selected.values() if r["extraction"] is not None
    ]
    if extracted and not recovering:
        raise ValueError("candidate is already extracted")
    if not recovering:
        if raw["state"] != "open" or raw.get("state_reason") == "reopened":
            raise ValueError("closed or reopened proposal cannot be extracted")
        if deadline(raw) <= datetime.now(UTC):
            raise ValueError(
                "expired proposal cannot be extracted; run improvement-expire"
            )
        approved_keys = {
            k
            for k, r in members(recorded, proposal).items()
            if human_state(r) == "approved" and r["extraction"] is None
        }
        if set(keys) != approved_keys:
            raise ValueError(
                f"extraction must include all approved candidates: missing={sorted(approved_keys - set(keys))}, extra={sorted(set(keys) - approved_keys)}"
            )
    for k, row in selected.items():
        validate_candidate(row)
        if human_state(row) != "approved":
            raise ValueError(
                f"I-{session.parent}-{k}: human execution approval is {human_state(row)}"
            )
        if row["decisions"][-1]["revision"] != row["revision"]:
            raise ValueError("candidate revision changed after approval")
    decision_ids = {k: r["decisions"][-1]["id"] for k, r in selected.items()}
    if receipt is None:
        draft = extraction_draft(session, source, selected)
        receipt = {
            "repo": session.gh.repo.casefold(),
            "parent": session.parent,
            "keys": keys,
            "proposal": proposal,
            "decision_ids": decision_ids,
            "creation": {"draft": draft, "attempted": False, "number": None},
        }
    elif receipt["decision_ids"] != decision_ids:
        raise ValueError(
            "approval changed during extraction; reconcile the fixed parent without granting execution"
        )
    rows = session.gh.issue_rows()
    matches = [row for row in rows if parse_extraction(row.get("body") or "") == source]
    # A candidate must not appear in any other extraction, including closed ones.
    for other in rows:
        origin = parse_extraction(other.get("body") or "")
        if (
            origin
            and origin.parent == source.parent
            and origin != source
            and set(origin.keys) & set(keys)
        ):
            raise ValueError("candidate already appears in another extracted parent")
    draft = receipt["creation"]["draft"]
    session.public(draft)
    created = create_issue_once(
        session.gh,
        draft=draft,
        receipt=receipt["creation"],
        matches=matches,
        save=lambda: save_receipt(session, name, receipt),
    )
    target = positive(created["number"])
    requirements = contract(draft["body"])
    for k in keys:
        row = candidates(session)[k]
        data = {
            "issue": target,
            "proposal": proposal,
            "decision_ids": decision_ids,
            "requirements": requirements,
        }
        if row["extraction"] is None:
            append(
                session,
                event(
                    ImprovementSource(session.parent, k),
                    "extracted",
                    row["revision"],
                    row["last"]["id"],
                    data,
                ),
            )
        elif row["extraction"]["data"] != data:
            raise ValueError("conflicting extraction record")
    refresh(session, raw, ended="提案終了", close="completed")
    return {
        "issue": target,
        "proposal": proposal,
        "candidates": [f"I-{session.parent}-{k}" for k in keys],
        "decision_ids": list(decision_ids.values()),
        "reused": bool(matches),
    }


def expire(gh: Any, store: Any, *, now: str | None, dry_run: bool) -> dict[str, Any]:
    """Return partial outcomes on failure; caller turns errors into nonzero status."""
    from .session import ParentSession

    current = _utc_timestamp(now, "now") if now else datetime.now(UTC)
    result: dict[str, Any] = {
        "closed": [],
        "pending": [],
        "errors": [],
        "dry_run": dry_run,
    }
    try:
        rows = gh.issue_rows()
        index: dict[Proposal, dict[str, Any]] = {}
        for raw in rows:
            source = parse_proposal(raw.get("body") or "")
            if source:
                if source in index:
                    raise ValueError("duplicate proposal provenance")
                index[source] = raw
        known = {source.parent for source in index}
        directory = (
            store.directory / "improvement-proposals" / fingerprint(gh.repo.casefold())
        )
        for path in directory.glob("proposal-*.json"):
            token = path.stem.removeprefix("proposal-")
            if not token.isdecimal():
                raise ValueError("invalid proposal receipt filename")
            known.add(positive(int(token)))
        for parent in sorted(known):
            session = ParentSession(gh, store, parent, None)
            index_proposals(session, rows, candidates(session))
            receipt = load_receipt(session, f"proposal-{parent}")
            if (
                receipt
                and receipt["creation"]["attempted"]
                and Proposal(parent, receipt["generation"]) not in index
            ):
                raise ValueError(
                    "proposal creation outcome/source is unknown; recover improvement-propose"
                )
        for source, raw in sorted(
            index.items(), key=lambda item: (item[0].parent, item[0].generation)
        ):
            session = ParentSession(gh, store, source.parent, None)
            recorded = candidates(session)
            index_proposals(session, rows, recorded)
            included = members(recorded, raw["number"])
            # An attempted extraction is reserved for explicit recovery, never expiry.
            receipt = load_receipt(session, f"extraction-{raw['number']}")
            if (
                any(r["extraction"] is not None for r in included.values())
                or (receipt and receipt["creation"]["attempted"])
                or raw["state"] != "open"
                or deadline(raw) > current
            ):
                continue
            if not included:
                raise ValueError(
                    f"proposal #{raw['number']} has no canonical membership; recover improvement-propose first"
                )
            result["pending"].append(raw["number"])
            if dry_run:
                continue
            body = (
                f"<!-- parent-improvement-expired:v1 proposal={raw['number']} -->\n\n期限切れのため提案を終了します。失効した候補:\n"
                + "\n".join(
                    f"- I-{source.parent}-{k}: "
                    + (
                        "承認済み・未取り出し（承認失効）"
                        if human_state(r) == "approved"
                        else "未取り出し"
                    )
                    for k, r in sorted(included.items())
                )
            )
            comments = gh.rows(f"repos/{gh.repo}/issues/{raw['number']}/comments")
            matches = [
                c
                for c in comments
                if (c.get("body") or "").startswith(body.splitlines()[0])
                and c.get("user", {}).get("login") == session.login
            ]
            if any(c["body"] != body for c in matches):
                raise ValueError("conflicting expiry comment")
            if not matches:
                gh.write(
                    "POST",
                    f"repos/{gh.repo}/issues/{raw['number']}/comments",
                    {"body": body},
                )
            refresh(session, raw, ended="期限切れ", close="not_planned")
            result["closed"].append(raw["number"])
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
    return result
