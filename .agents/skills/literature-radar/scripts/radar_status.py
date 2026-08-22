#!/usr/bin/env python
"""Normalize and validate Literature Radar daily status files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

QUOTA_MODES = {"enforced", "historical_backfill"}


class StatusError(ValueError):
    """Raised when a Literature Radar status file is invalid."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatusError(f"{path}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise StatusError(f"{path}: top-level JSON value must be an object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_config(repo_root: Path) -> dict[str, Any]:
    return read_json(repo_root / "knowledge/literature/config.json")


def candidate_records(repo_root: Path) -> list[dict[str, Any]]:
    directory = repo_root / "knowledge/literature/candidates"
    if not directory.exists():
        return []
    return [read_json(path) for path in sorted(directory.glob("paper-*.json"))]


def local_date(value: object, timezone: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StatusError("candidate first_seen must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise StatusError(f"invalid candidate first_seen: {value!r}") from exc
    if parsed.tzinfo is None:
        raise StatusError("candidate first_seen must include an explicit offset")
    return parsed.astimezone(ZoneInfo(timezone)).date().isoformat()


def record_has_collector(record: dict[str, Any], collector_id: str) -> bool:
    return any(
        isinstance(item, dict) and item.get("collector_id") == collector_id
        for item in record.get("discoveries", [])
    )


def record_topics(record: dict[str, Any]) -> set[str]:
    topics: set[str] = set()
    for discovery in record.get("discoveries", []):
        if not isinstance(discovery, dict):
            continue
        screening = discovery.get("screening")
        if not isinstance(screening, dict):
            continue
        topic = screening.get("topic")
        if isinstance(topic, str) and topic:
            topics.add(topic)
    return topics


def normalize_status(repo_root: Path, date: str) -> Path:
    path = repo_root / "knowledge/literature/status" / f"{date}.json"
    status = read_json(path)
    status.setdefault("quota_mode", "enforced")
    status.setdefault("quota_note", None)
    write_json(path, status)
    return path


def _quota_errors(
    *,
    name: str,
    actual: object,
    expected_accepted: int,
    configured_limit: int,
    mode: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(actual, dict):
        return [f"{name} must be an object"]

    accepted = actual.get("accepted")
    limit = actual.get("limit")
    remaining = actual.get("remaining")
    if accepted != expected_accepted:
        errors.append(
            f"{name}.accepted must be {expected_accepted}, got {accepted!r}"
        )
    if limit != configured_limit:
        errors.append(f"{name}.limit must be {configured_limit}, got {limit!r}")
    expected_remaining = max(0, configured_limit - expected_accepted)
    if remaining != expected_remaining:
        errors.append(
            f"{name}.remaining must be {expected_remaining}, got {remaining!r}"
        )
    if mode == "enforced" and expected_accepted > configured_limit:
        errors.append(
            f"{name} exceeds enforced quota "
            f"({expected_accepted}/{configured_limit})"
        )
    return errors


def _declared_quota_errors(
    *,
    name: str,
    actual: object,
    configured_limit: int,
) -> list[str]:
    if not isinstance(actual, dict):
        return [f"{name} must be an object"]
    accepted = actual.get("accepted")
    if (
        not isinstance(accepted, int)
        or isinstance(accepted, bool)
        or accepted < 0
    ):
        return [f"{name}.accepted must be a non-negative integer"]
    return _quota_errors(
        name=name,
        actual=actual,
        expected_accepted=accepted,
        configured_limit=configured_limit,
        mode="historical_backfill",
    )


def validate_status(
    path: Path,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    validate_current_backlog: bool = True,
) -> list[str]:
    errors: list[str] = []
    try:
        status = read_json(path)
    except StatusError as exc:
        return [str(exc)]

    date = status.get("date")
    if date != path.stem:
        errors.append(f"{path}: date must match filename {path.stem!r}")
    if not isinstance(date, str):
        return errors + [f"{path}: date must be a string"]

    mode = status.get("quota_mode", "enforced")
    note = status.get("quota_note")
    if mode not in QUOTA_MODES:
        errors.append(f"{path}: quota_mode must be one of {sorted(QUOTA_MODES)}")
        mode = "enforced"
    if mode == "historical_backfill" and not (
        isinstance(note, str) and note.strip()
    ):
        errors.append(
            f"{path}: historical_backfill requires a non-empty quota_note"
        )
    if note is not None and not isinstance(note, str):
        errors.append(f"{path}: quota_note must be a string or null")

    ingestion = status.get("ingestion")
    if not isinstance(ingestion, dict):
        return errors + [f"{path}: ingestion must be an object"]

    timezone = str(config.get("timezone", "Asia/Tokyo"))
    same_day: list[dict[str, Any]] = []
    for record in records:
        try:
            if local_date(record.get("first_seen"), timezone) == date:
                same_day.append(record)
        except StatusError as exc:
            errors.append(f"{path}: {record.get('id')}: {exc}")

    settings = config["ingestion"]
    daily_limit = int(settings["max_candidates_total_per_day"])
    accepted = len(same_day)
    if ingestion.get("accepted_candidates") != accepted:
        errors.append(
            f"{path}: accepted_candidates must be {accepted}, "
            f"got {ingestion.get('accepted_candidates')!r}"
        )
    if ingestion.get("daily_limit") != daily_limit:
        errors.append(
            f"{path}: daily_limit must be {daily_limit}, "
            f"got {ingestion.get('daily_limit')!r}"
        )
    if mode == "enforced" and accepted > daily_limit:
        errors.append(
            f"{path}: daily quota exceeded under enforced mode "
            f"({accepted}/{daily_limit})"
        )

    open_limit = int(settings["max_open_candidates"])
    open_candidates = ingestion.get("open_candidates")
    if (
        not isinstance(open_candidates, int)
        or isinstance(open_candidates, bool)
        or open_candidates < 0
    ):
        errors.append(f"{path}: open_candidates must be a non-negative integer")
    else:
        if validate_current_backlog:
            open_count = sum(
                record.get("state") == "inbox" for record in records
            )
            if open_candidates != open_count:
                errors.append(
                    f"{path}: open_candidates must be {open_count}, "
                    f"got {open_candidates!r}"
                )
        if open_candidates > open_limit:
            errors.append(
                f"{path}: open candidate backlog exceeds {open_limit} "
                f"(snapshot {open_candidates})"
            )
    if ingestion.get("open_limit") != open_limit:
        errors.append(
            f"{path}: open_limit must be {open_limit}, "
            f"got {ingestion.get('open_limit')!r}"
        )

    collector_limit = int(settings["max_candidates_per_collector_per_day"])
    collectors = ingestion.get("collectors")
    if not isinstance(collectors, dict):
        errors.append(f"{path}: collectors must be an object")
    else:
        expected_collectors = set(config["collectors"])
        if set(collectors) != expected_collectors:
            errors.append(
                f"{path}: collector keys must be {sorted(expected_collectors)}"
            )
        for collector_id in sorted(expected_collectors):
            count = sum(
                record_has_collector(record, collector_id) for record in same_day
            )
            errors.extend(
                f"{path}: {error}"
                for error in _quota_errors(
                    name=f"collectors.{collector_id}",
                    actual=collectors.get(collector_id),
                    expected_accepted=count,
                    configured_limit=collector_limit,
                    mode=mode,
                )
            )

    topic_limit = int(settings["max_candidates_per_topic_per_day"])
    expected_topics = {
        str(topic)
        for collector in config["collectors"].values()
        for topic in collector.get("topics", [])
    }
    topics = ingestion.get("topics")
    if not isinstance(topics, dict):
        errors.append(f"{path}: topics must be an object")
        return errors
    if set(topics) != expected_topics:
        errors.append(f"{path}: topic keys must be {sorted(expected_topics)}")

    # Pre-hardening records did not require screening.topic. A documented
    # historical backfill cannot reconstruct topic counts from canonical
    # records, but its declared snapshot must still use the configured keys,
    # limits and remaining-count arithmetic.
    if mode == "historical_backfill":
        for topic in sorted(expected_topics):
            errors.extend(
                f"{path}: {error}"
                for error in _declared_quota_errors(
                    name=f"topics.{topic}",
                    actual=topics.get(topic),
                    configured_limit=topic_limit,
                )
            )
        return errors

    for topic in sorted(expected_topics):
        count = sum(topic in record_topics(record) for record in same_day)
        errors.extend(
            f"{path}: {error}"
            for error in _quota_errors(
                name=f"topics.{topic}",
                actual=topics.get(topic),
                expected_accepted=count,
                configured_limit=topic_limit,
                mode=mode,
            )
        )
    return errors


def validate_repository(repo_root: Path) -> list[str]:
    config = load_config(repo_root)
    records = candidate_records(repo_root)
    directory = repo_root / "knowledge/literature/status"
    if not directory.exists():
        return []

    paths = sorted(directory.glob("????-??-??.json"))
    if not paths:
        return []

    # open_candidates is a point-in-time global backlog snapshot. Canonical
    # candidate records retain only their current state, so an older snapshot
    # cannot be reconstructed from the present tree. Only the newest daily
    # status is expected to match the current canonical inbox exactly.
    current_path = paths[-1]
    errors: list[str] = []
    for path in paths:
        errors.extend(
            validate_status(
                path,
                records,
                config,
                validate_current_backlog=path == current_path,
            )
        )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    normalize = commands.add_parser("normalize")
    normalize.add_argument("--repo-root", type=Path, default=Path.cwd())
    normalize.add_argument("--date", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.command == "normalize":
            print(normalize_status(repo_root, args.date))
            return 0
        errors = validate_repository(repo_root)
        for error in errors:
            print(f"ERROR: {error}")
        print(f"{len(errors)} literature status error(s).")
        return 1 if errors else 0
    except StatusError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
