"""Bind parent CLI choices and source references before publishing a request."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from lib.review_settings import (
    choose,
    load_policy,
)

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
    parser.add_argument("--review-config-file", type=Path)


def selection(
    session: ParentSession | None, args: argparse.Namespace
) -> dict[str, Any]:
    policy = load_policy(args.review_config_file)
    if policy != load_policy(None):
        raise ValueError("parent review requires the configured Sol policy")
    return choose(
        policy,
        model=args.review_model,
        effort=args.review_effort,
        reason=Path(args.model_reason_file).read_text(encoding="utf-8")
        if args.model_reason_file
        else None,
        specified_by=args.review_model_source,
        human_model=args.human_review_model,
        human_effort=args.human_review_effort,
        human_reference=args.human_model_reference,
    )
