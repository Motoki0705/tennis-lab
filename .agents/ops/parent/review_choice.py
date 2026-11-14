"""Bind parent CLI choices and source references before publishing a request."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from lib.review_settings import (
    choose,
    load_policy,
    read_object,
    run_id,
    validate_selection,
)

from .policy import digest
from .session import ParentSession


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--review-model")
    parser.add_argument("--review-effort")
    parser.add_argument("--model-reason-file")
    parser.add_argument(
        "--review-model-source", choices=("parent_choice", "human_instruction")
    )
    parser.add_argument("--human-review-model")
    parser.add_argument("--human-review-effort")
    parser.add_argument("--human-model-reference")
    parser.add_argument("--recover-source-run")
    parser.add_argument("--handoff-from-run")
    parser.add_argument("--handoff-scope-file")
    parser.add_argument("--review-config-file", type=Path)


def source(
    session: ParentSession, identifier: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    run_id(identifier)
    registered = next(
        (
            r["request"]
            for r in session.row["reviews"]
            if r["request"]["review_run_id"] == identifier
        ),
        None,
    )
    if registered is None:
        raise ValueError("source review is not registered with this parent")
    directory = session.store.directory / "reviews" / identifier
    request = read_object(directory / "request.json")
    if (
        request != registered
        or request["repo"].casefold() != session.gh.repo.casefold()
        or request["parent_issue"] != session.parent
    ):
        raise ValueError("source review request differs from registry")
    result = (
        read_object(directory / "result.json")
        if (directory / "result.json").exists()
        else None
    )
    if result is not None and (
        result.get("request") != request
        or result.get("review_run_id") != identifier
        or result.get("request_sha256") != digest(request)
    ):
        raise ValueError("source review result differs from its request")
    return request, result


def candidate_records(
    session: ParentSession,
    request: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    adopting: bool = False,
) -> list[dict[str, Any]]:
    # The review modules also serve standalone scripts; match review-inspect's
    # module search paths only at this explicit review entry point.
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "review", root / "lib"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from review.review_recovery import handoff_candidates

    with session.gh.io_scope():
        records: list[dict[str, Any]] = handoff_candidates(
            session.store.directory, request, result, adopting=adopting
        )
    return records


def selection(
    session: ParentSession, args: argparse.Namespace
) -> dict[str, Any] | None:
    names = (
        "review_model",
        "review_effort",
        "model_reason_file",
        "review_model_source",
        "human_review_model",
        "human_review_effort",
        "human_model_reference",
        "recover_source_run",
        "handoff_from_run",
        "handoff_scope_file",
        "review_config_file",
    )
    if getattr(args, "run_id", None) and all(
        getattr(args, name) is None for name in names
    ):
        return None
    if args.recover_source_run is not None:
        if any(
            getattr(args, name) is not None
            for name in names
            if name != "recover_source_run"
        ):
            raise ValueError(
                "recovery source cannot be combined with a new model choice"
            )
        request, result = source(session, args.recover_source_run)
        return validate_selection(
            {
                "kind": "recover",
                "source_run": args.recover_source_run,
                "source_request_sha256": digest(request),
                "candidates": candidate_records(
                    session, request, result, adopting=True
                ),
            }
        )
    handoff = None
    if bool(args.handoff_from_run) != bool(args.handoff_scope_file):
        raise ValueError("handoff requires both source run and scope file")
    if args.handoff_from_run:
        request, result = source(session, args.handoff_from_run)
        if result is None:
            raise ValueError("handoff source has no saved result")
        handoff = {
            "source_run": args.handoff_from_run,
            "source_request_sha256": digest(request),
            "source_result_sha256": digest(result),
            "scope": Path(args.handoff_scope_file).read_text(encoding="utf-8"),
            "candidates": candidate_records(session, request, result),
        }
    return choose(
        load_policy(args.review_config_file),
        model=args.review_model,
        effort=args.review_effort,
        reason=Path(args.model_reason_file).read_text(encoding="utf-8")
        if args.model_reason_file
        else None,
        specified_by=args.review_model_source,
        human_model=args.human_review_model,
        human_effort=args.human_review_effort,
        human_reference=args.human_model_reference,
        handoff=handoff,
    )


def verify_references(
    session: ParentSession, choice: dict[str, Any], prs: list[dict[str, Any]]
) -> None:
    validate_selection(choice)
    reference = choice if choice["kind"] == "recover" else choice["handoff"]
    if reference is None:
        return
    request, result = source(session, reference["source_run"])
    if reference["source_request_sha256"] != digest(request):
        raise ValueError("source review request changed")
    if choice["kind"] == "recover":
        if prs != request["prs"]:
            raise ValueError("recovery requires unchanged source snapshots")
        if reference["candidates"] != candidate_records(
            session, request, result, adopting=True
        ):
            raise ValueError("recovery inherited candidates changed")
    elif (
        result is None
        or reference["source_result_sha256"] != digest(result)
        or not {p["number"] for p in prs} <= {p["number"] for p in request["prs"]}
        or reference["candidates"] != candidate_records(session, request, result)
    ):
        raise ValueError("handoff source result/PR correspondence changed")
