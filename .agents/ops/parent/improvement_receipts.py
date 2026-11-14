"""Immutable creation receipts and extraction bindings independent of Issue prose."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from issues.improvement import fields, fingerprint, key, positive
from issues.improvement_proposals import (
    Extraction,
    Proposal,
    parse_extraction,
    parse_proposal,
)
from worker_logs.journal import atomic, decode

if TYPE_CHECKING:
    from .session import ParentSession


def receipt_directory(directory: Path, repo: str) -> Path:
    repo_key: str = fingerprint(repo.casefold())
    return directory / "improvement-proposals" / repo_key


def receipt_path(session: ParentSession, name: str) -> Path:
    return (
        receipt_directory(Path(session.store.directory), session.gh.repo)
        / f"{name}.json"
    )


def load_receipt(session: ParentSession, name: str) -> dict[str, Any] | None:
    return read_receipt(session.store.directory, session.gh.repo, name)


def read_receipt(directory: Path, repo: str, name: str) -> dict[str, Any] | None:
    path = receipt_directory(directory, repo) / f"{name}.json"
    if path.is_symlink():
        raise ValueError("improvement receipt must not be a symlink")
    if not path.exists():
        return None
    value = decode(path.read_bytes())
    if not isinstance(value, dict) or value.get("repo") != repo.casefold():
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


def extraction_index(directory: Path, repo: str) -> dict[str, Any]:
    """Read every durable binding; an uncertain POST must be recovered first.

    Receipts identify the target even before registration/canonical event writes.
    They never replace the live candidate approval checks.
    """
    root = receipt_directory(directory, repo)
    if root.is_symlink():
        raise ValueError("improvement receipt directory must not be a symlink")
    try:
        paths = sorted(root.iterdir())
    except FileNotFoundError:
        return {}
    result: dict[str, Any] = {}
    for path in paths:
        if not path.name.startswith("extraction-") or path.suffix != ".json":
            continue
        receipt = read_receipt(directory, repo, path.stem)
        if receipt is None:
            raise ValueError("extraction receipt disappeared while reading its index")
        creation = receipt["creation"]
        target = creation["number"]
        if target is None:
            if creation["attempted"]:
                raise ValueError(
                    "extraction outcome is unknown; recover improvement-extract before execution"
                )
            continue
        if not creation["attempted"]:
            raise ValueError("extraction receipt target has no creation attempt")
        binding = {
            k: receipt[k] for k in ("parent", "proposal", "keys", "decision_ids")
        }
        if str(target) in result:
            raise ValueError("duplicate extraction target in receipt index")
        result[str(target)] = binding
    return result


def bound_extraction(
    body: str, target: int, index: Mapping[str, Any]
) -> Extraction | None:
    """Fail closed when editable provenance differs from the numbered receipt."""
    source = parse_extraction(body)
    binding = index.get(str(target))
    if binding is None:
        if source is not None:
            raise ValueError("extraction provenance has no receipt in the target index")
        return None
    expected = Extraction(
        binding["parent"], binding["proposal"], tuple(binding["keys"])
    )
    if source != expected:
        raise ValueError("extraction provenance differs from the target receipt index")
    return expected
