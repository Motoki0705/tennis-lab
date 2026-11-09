"""The public record and the addition-only merge exception."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from .journal import decode, require
from .safety import Guard

DIRECTORY = ".agents/logs/parents"
FILENAMES = {"instruction.md", "final.md", "meta.json"}
META_KEYS = {
    "schema_version",
    "repo",
    "parent",
    "issue",
    "delegation_key",
    "turn",
    "decision",
    "sources",
    "instruction",
    "worker",
    "runtime",
    "artifacts",
    "files",
    "sanitization",
    "started_at",
    "finished_at",
    "captured_at",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def fields(value: Any, expected: set[str]) -> None:
    require(isinstance(value, dict) and set(value) == expected, "record_schema")


def integer(value: Any) -> bool:
    return type(value) is int and value > 0


def digest(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(r"[a-f0-9]{" + str(length) + "}", value)
    )


def sanitization(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "record_sanitization")
    version = value.get("version")
    require(type(version) is int and version in {1, 2}, "record_sanitization")
    expected = {"version", "instruction", "final", "additional"}
    if version == 2:
        expected.add("local_paths")
    require(set(value) == expected, "record_sanitization")
    require(
        all(
            type(value[k]) is int and value[k] >= 0
            for k in ("instruction", "final", "additional")
        ),
        "record_sanitization",
    )
    if version == 2:
        counts = value["local_paths"]
        require(
            isinstance(counts, dict)
            and bool(counts)
            and set(counts) <= {"posix", "home", "windows", "unc"}
            and all(type(n) is int and n > 0 for n in counts.values())
            and sum(counts.values())
            <= sum(value[k] for k in ("instruction", "final", "additional")),
            "record_sanitization",
        )
    return dict(value)


def path(meta: dict[str, Any]) -> str:
    require(integer(meta.get("parent")) and integer(meta.get("issue")), "record_issue")
    require(
        bool(re.fullmatch(r"d-[a-f0-9]{32}", str(meta.get("delegation_key")))),
        "record_key",
    )
    turn = meta["turn"]
    fields(turn, {"sequence", "kind", "phase", "previous_sequence"})
    require(integer(turn["sequence"]), "record_turn")
    require(turn["kind"] in {"initial", "resume"}, "record_turn")
    require(turn["phase"] in {"planning", "implementing"}, "record_phase")
    previous = turn["previous_sequence"]
    require(
        previous is None or integer(previous) and previous < turn["sequence"],
        "record_turn",
    )
    return (
        f"{DIRECTORY}/parent-{meta['parent']}/child-{meta['issue']}/"
        f"{meta['delegation_key']}/{turn['sequence']:04d}-{turn['kind']}"
    )


def validate(files: dict[str, bytes], guard: Guard | None = None) -> dict[str, Any]:
    require(
        set(files) == FILENAMES and all(isinstance(v, bytes) for v in files.values()),
        "record_files",
    )
    meta = decode(files["meta.json"])
    fields(meta, META_KEYS)
    require(
        type(meta["schema_version"]) is int and meta["schema_version"] == 1,
        "record_version",
    )
    repo = meta["repo"]
    require(
        isinstance(repo, str)
        and bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)),
        "record_repository",
    )
    path(meta)
    for name in ("instruction.md", "final.md"):
        require(bool(files[name].decode("utf-8").strip()), "empty_text")
    fields(meta["files"], {"instruction.md", "final.md"})
    require(meta["files"] == {n: sha(files[n]) for n in meta["files"]}, "record_digest")
    decision = meta["decision"]
    if decision is not None:
        fields(decision, {"action", "source_sequence"})
        require(
            decision["action"] in {"approve", "investigate", "reject"},
            "record_decision",
        )
        require(
            integer(decision["source_sequence"])
            and decision["source_sequence"] < meta["turn"]["sequence"],
            "record_decision",
        )
        require(
            meta["turn"]["phase"]
            == ("implementing" if decision["action"] == "approve" else "planning"),
            "record_decision",
        )
    fields(meta["sources"], {"standard_instruction", "issue"})
    standard = meta["sources"]["standard_instruction"]
    fields(standard, {"commit", "path", "blob"})
    require(
        digest(standard["commit"], 40) and digest(standard["blob"], 40), "record_source"
    )
    require(standard["path"] == ".agents/ops/parent/worker_prompt.md", "record_source")
    issue = meta["sources"]["issue"]
    fields(issue, {"url", "body_sha256"})
    require(
        issue["url"] == f"https://github.com/{repo}/issues/{meta['issue']}"
        and digest(issue["body_sha256"]),
        "record_source",
    )
    fields(meta["instruction"], {"kind", "additional_instructions"})
    require(meta["instruction"]["kind"] in {"task", "answer"}, "record_instruction")
    require(
        isinstance(meta["instruction"]["additional_instructions"], list),
        "record_instruction",
    )
    require(
        all(
            isinstance(s, str) and s.strip()
            for s in meta["instruction"]["additional_instructions"]
        ),
        "record_instruction",
    )
    worker = meta["worker"]
    fields(worker, {"agent", "selected", "effective"})
    require(worker["agent"] == "codex", "record_worker")
    fields(worker["selected"], {"model", "effort"})
    require(
        isinstance(worker["selected"]["model"], str)
        and bool(re.fullmatch(r"[A-Za-z0-9_.-]+", worker["selected"]["model"]))
        and worker["selected"]["effort"] == "max",
        "record_worker",
    )
    fields(worker["effective"], {"model", "effort"})
    for key, value in worker["effective"].items():
        fields(value, {"status", "value"})
        require(
            value
            in (
                {"status": "observed", "value": worker["selected"][key]},
                {"status": "unavailable", "value": None},
            ),
            "record_worker",
        )
    require(
        meta["runtime"] == {"result": "completed", "turn_end_reason": "completed"},
        "record_runtime",
    )
    fields(meta["artifacts"], {"prs", "commits"})
    require(
        isinstance(meta["artifacts"]["prs"], list)
        and isinstance(meta["artifacts"]["commits"], list),
        "record_artifact",
    )
    require(
        all(integer(n) for n in meta["artifacts"]["prs"])
        and all(digest(s, 40) for s in meta["artifacts"]["commits"]),
        "record_artifact",
    )
    sanitization(meta["sanitization"])
    for key in ("started_at", "finished_at", "captured_at"):
        value = datetime.fromisoformat(meta[key])
        require(value.tzinfo is not None, "record_time")
    checker = guard or Guard()
    checker.check(meta)
    checker.check([n for n in files])
    checker.check([files[n].decode("utf-8") for n in ("instruction.md", "final.md")])
    checker.check(path(meta))
    return meta


def additions(raw: bytes, expected: set[str]) -> None:
    """Check raw -z --no-renames output, including source paths and Git modes."""
    entries = raw.split(b"\0")
    require(entries[-1:] == [b""] and len(entries) % 2 == 1, "unsafe_diff")
    found = []
    for i in range(0, len(entries) - 1, 2):
        header = entries[i].decode("ascii").split()
        require(
            len(header) == 5
            and header[0] == ":000000"
            and header[1] == "100644"
            and header[4] == "A",
            "unsafe_diff",
        )
        found.append(entries[i + 1].decode("utf-8"))
    require(len(found) == len(set(found)) and set(found) == expected, "unsafe_diff")


def identity(pr: dict[str, Any], row: dict[str, Any], repo: str) -> None:
    source = row["source"]
    require(
        pr.get("base", {}).get("ref") == "main"
        and pr.get("base", {}).get("repo", {}).get("full_name", "").casefold()
        == repo.casefold()
        and pr.get("head", {}).get("repo", {}).get("full_name", "").casefold()
        == repo.casefold()
        and pr.get("head", {}).get("ref") == row["branch"]
        and pr.get("head", {}).get("sha") == row["head"]
        and pr.get("draft") is False
        and pr.get("body") == row["pr_body"]
        and pr.get("html_url") == f"https://github.com/{repo}/pull/{pr.get('number')}"
        and integer(pr.get("number"))
        and source["repo"].casefold() == repo.casefold(),
        "pr_identity_changed",
    )
