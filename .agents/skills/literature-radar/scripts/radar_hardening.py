#!/usr/bin/env python
"""Harden scheduled literature ingestion while preserving existing records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import radar_ingest as legacy

EVIDENCE_RANK = {
    "abstract": 0,
    "fulltext": 1,
    "fulltext-code": 2,
    "fulltext-code-data": 3,
}
SCORE_KEYS = (
    "task_fit",
    "repo_fit",
    "evidence_quality",
    "experiment_quality",
    "adoption_feasibility",
)
SAFE_DIGEST_PREAMBLE = (
    "# Literature Radar \u2014 {date}\n\n"
    "\u3053\u306e\u65e5\u6b21\u30c0\u30a4\u30b8\u30a7\u30b9\u30c8"
    "\u306e\u81ea\u52d5\u53ce\u96c6\u533a\u9593\u306f"
    "GitHub Actions\u304c\u66f4\u65b0\u3057\u307e\u3059\u3002\n"
    "\u65e5\u6b21curator\u306f\u81ea\u52d5\u533a\u9593"
    "\u306e\u5916\u5074\u3060\u3051\u3092\u7de8\u96c6"
    "\u3057\u3066\u304f\u3060\u3055\u3044\u3002\n\n"
)
LEGACY_BROKEN_PREAMBLE = (
    "\u3053\u306e\u65e5\u6b21\u30c0\u30a4\u30b8\u30a7\u30b9\u30c8"
    "\u306e `"
)


def _normalise_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _normalise_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            parsed.path.rstrip("/"),
            "",
            "",
        )
    )


def _arxiv_from_datacite_doi(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    doi = legacy._normalise_doi(value)
    match = re.fullmatch(
        r"10\.48550/arxiv\.(.+)",
        doi,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return legacy._normalise_arxiv(match.group(1))


def _publisher_doi(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if _arxiv_from_datacite_doi(value) is not None:
        return None
    return legacy._normalise_doi(value)


def _paper_identifiers(
    paper: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    identifiers = paper.get("identifiers")
    if not isinstance(identifiers, dict):
        return None, None, None

    doi = _publisher_doi(identifiers.get("doi"))
    raw_arxiv = identifiers.get("arxiv")
    if raw_arxiv is None:
        raw_arxiv = identifiers.get("arXiv")
    if isinstance(raw_arxiv, str) and raw_arxiv.strip():
        arxiv = legacy._normalise_arxiv(raw_arxiv)
    else:
        arxiv = _arxiv_from_datacite_doi(identifiers.get("doi"))

    raw_openreview = identifiers.get("openreview")
    if isinstance(raw_openreview, str) and raw_openreview.strip():
        openreview = legacy._normalise_openreview(raw_openreview)
    else:
        openreview = None
    return doi, arxiv, openreview


def canonical_paper_id(payload: dict[str, Any]) -> str:
    paper = payload["paper"]
    doi, arxiv, openreview = _paper_identifiers(paper)
    if doi:
        return f"paper-doi-{legacy._slug(doi)}"
    if arxiv:
        return f"paper-arxiv-{legacy._slug(arxiv)}"
    if openreview:
        return f"paper-openreview-{legacy._slug(openreview)}"

    title = _normalise_title(str(paper["title"]))
    year = int(paper["year"])
    digest = hashlib.sha256(
        f"{title}\n{year}".encode("utf-8")
    ).hexdigest()[:16]
    return f"paper-title-{year}-{digest}"


def paper_aliases(paper: dict[str, Any]) -> set[str]:
    doi, arxiv, openreview = _paper_identifiers(paper)
    aliases: set[str] = set()
    if doi:
        aliases.add(f"doi:{doi}")
    if arxiv:
        aliases.add(f"arxiv:{arxiv}")
    if openreview:
        aliases.add(f"openreview:{openreview}")

    title = _normalise_title(str(paper.get("title", "")))
    year = paper.get("year")
    if title and isinstance(year, int) and not isinstance(year, bool):
        aliases.add(f"title:{year}:{title}")

    urls = paper.get("urls")
    primary = urls.get("primary") if isinstance(urls, dict) else None
    if isinstance(primary, str) and primary.strip():
        aliases.add(f"url:{_normalise_url(primary)}")
    return aliases


def _sanitise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(payload)
    identifiers = clean["paper"]["identifiers"]
    mixed_case_arxiv = identifiers.pop("arXiv", None)
    if not identifiers.get("arxiv") and mixed_case_arxiv:
        identifiers["arxiv"] = mixed_case_arxiv

    arxiv_alias = _arxiv_from_datacite_doi(identifiers.get("doi"))
    if arxiv_alias is not None:
        identifiers["doi"] = None
        if not identifiers.get("arxiv"):
            identifiers["arxiv"] = arxiv_alias
    return clean


def _validate_score_breakdown(
    screening: dict[str, Any],
    config: dict[str, Any],
) -> None:
    breakdown = screening.get("score_breakdown")
    if not isinstance(breakdown, dict):
        raise legacy.CandidateError(
            "screening.score_breakdown must be an object"
        )
    if set(breakdown) != set(SCORE_KEYS):
        raise legacy.CandidateError(
            "screening.score_breakdown must contain exactly "
            f"{list(SCORE_KEYS)}"
        )

    maxima = config["ingestion"]["score_weights"]
    for key in SCORE_KEYS:
        value = breakdown.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= int(maxima[key])
        ):
            raise legacy.CandidateError(
                f"screening.score_breakdown.{key} must be "
                f"an integer from 0 to {maxima[key]}"
            )

    relevance = int(screening["relevance_score"])
    total = sum(int(breakdown[key]) for key in SCORE_KEYS)
    if total != relevance:
        raise legacy.CandidateError(
            "screening.relevance_score must equal the "
            "score_breakdown sum"
        )


def _validate_evidence(
    payload: dict[str, Any],
    config: dict[str, Any],
) -> None:
    screening = payload["screening"]
    evidence = str(screening["evidence_level"])
    minimum = str(
        config["ingestion"].get(
            "minimum_evidence_level",
            "fulltext",
        )
    )
    if EVIDENCE_RANK[evidence] < EVIDENCE_RANK[minimum]:
        raise legacy.CandidateError(
            f"screening.evidence_level {evidence!r} is below "
            f"minimum {minimum!r}"
        )

    cap = int(
        config["ingestion"]["evidence_score_caps"][evidence]
    )
    evidence_score = int(
        screening["score_breakdown"]["evidence_quality"]
    )
    if evidence_score > cap:
        raise legacy.CandidateError(
            f"evidence_quality exceeds {evidence!r} cap {cap}"
        )

    allowed_kinds = set(config.get("allowed_source_kinds") or [])
    source_kinds = {
        str(source.get("kind"))
        for source in payload["sources"]
        if isinstance(source, dict)
    }
    unsupported = source_kinds - allowed_kinds
    if unsupported:
        raise legacy.CandidateError(
            f"sources contain unsupported kinds: {sorted(unsupported)}"
        )
    if evidence in {"fulltext-code", "fulltext-code-data"}:
        if "code" not in source_kinds:
            raise legacy.CandidateError(
                f"{evidence} requires an official code source"
            )
    if evidence == "fulltext-code-data":
        if "dataset" not in source_kinds:
            raise legacy.CandidateError(
                "fulltext-code-data requires an official dataset source"
            )


def validate_hardened_candidate(
    payload: dict[str, Any],
    repo_root: Path,
    config: dict[str, Any],
    expected_collector: str | None = None,
) -> None:
    legacy.validate_raw_candidate(
        payload,
        repo_root,
        config,
        expected_collector,
    )
    collector_id = str(payload["collector_id"])
    collector = config["collectors"][collector_id]
    screening = payload["screening"]

    tasks = {str(item) for item in screening.get("tasks", [])}
    allowed_tasks = {
        str(item) for item in collector.get("tasks", [])
    }
    if not tasks <= allowed_tasks:
        raise legacy.CandidateError(
            f"screening.tasks exceed collector {collector_id!r} "
            f"responsibility: {sorted(tasks - allowed_tasks)}"
        )

    topic = screening.get("topic")
    if topic not in collector.get("topics", []):
        raise legacy.CandidateError(
            f"screening.topic {topic!r} is not configured "
            f"for collector {collector_id!r}"
        )

    _validate_score_breakdown(screening, config)
    _validate_evidence(payload, config)


def record_date(
    record: dict[str, Any],
    timezone: str,
) -> str:
    return legacy._date_of_iso(
        str(record["first_seen"]),
        timezone,
    )


def record_has_collector(
    record: dict[str, Any],
    collector_id: str,
) -> bool:
    return any(
        isinstance(item, dict)
        and item.get("collector_id") == collector_id
        for item in record.get("discoveries", [])
    )


def record_topics(record: dict[str, Any]) -> set[str]:
    topics: set[str] = set()
    for item in record.get("discoveries", []):
        if not isinstance(item, dict):
            continue
        screening = item.get("screening")
        if not isinstance(screening, dict):
            continue
        topic = screening.get("topic")
        if isinstance(topic, str) and topic:
            topics.add(topic)
    return topics


def quota_allows(
    records: list[dict[str, Any]],
    payload: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, str]:
    settings = config["ingestion"]
    timezone = str(config.get("timezone", "Asia/Tokyo"))
    date = legacy.local_date(payload, config)
    same_day = [
        record
        for record in records
        if record_date(record, timezone) == date
    ]

    collector_id = str(payload["collector_id"])
    collector_count = sum(
        record_has_collector(record, collector_id)
        for record in same_day
    )
    collector_limit = int(
        settings["max_candidates_per_collector_per_day"]
    )
    if collector_count >= collector_limit:
        return (
            False,
            "collector daily quota reached "
            f"({collector_count}/{collector_limit})",
        )

    topic = str(payload["screening"]["topic"])
    topic_count = sum(
        topic in record_topics(record)
        for record in same_day
    )
    topic_limit = int(
        settings["max_candidates_per_topic_per_day"]
    )
    if topic_count >= topic_limit:
        return (
            False,
            "topic daily quota reached "
            f"({topic_count}/{topic_limit}) for {topic}",
        )

    daily_limit = int(settings["max_candidates_total_per_day"])
    if len(same_day) >= daily_limit:
        return (
            False,
            "global daily quota reached "
            f"({len(same_day)}/{daily_limit})",
        )

    open_count = sum(
        record.get("state") == "inbox"
        for record in records
    )
    open_limit = int(settings["max_open_candidates"])
    if open_count >= open_limit:
        return (
            False,
            "open candidate backlog limit reached "
            f"({open_count}/{open_limit})",
        )
    return True, ""


def _find_local_match(
    records: Iterable[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    incoming = paper_aliases(payload["paper"])
    matches = [
        record
        for record in records
        if incoming & paper_aliases(record["paper"])
    ]
    if len(matches) > 1:
        ids = [str(item["id"]) for item in matches]
        raise legacy.CandidateError(
            f"candidate aliases match multiple records: {ids}"
        )
    return matches[0] if matches else None


def aliases_in_git_refs(
    repo_root: Path,
    ref_prefix: str | None,
    target_date: str,
    window_days: int,
) -> set[str]:
    if not ref_prefix:
        return set()

    refs = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname)",
            ref_prefix,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    target = datetime.strptime(
        target_date,
        "%Y-%m-%d",
    ).date()
    aliases: set[str] = set()

    for ref in (
        line.strip()
        for line in refs.splitlines()
        if line.strip()
    ):
        try:
            branch_date = datetime.strptime(
                ref.rsplit("/", 1)[-1],
                "%Y-%m-%d",
            ).date()
        except ValueError:
            continue
        if abs((branch_date - target).days) > window_days:
            continue

        paths = subprocess.run(
            [
                "git",
                "ls-tree",
                "-r",
                "--name-only",
                ref,
                "--",
                "knowledge/literature/candidates",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for path in (
            line.strip()
            for line in paths.splitlines()
            if line.strip().endswith(".json")
        ):
            try:
                content = subprocess.run(
                    ["git", "show", f"{ref}:{path}"],
                    cwd=repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                record = json.loads(content)
            except (
                subprocess.CalledProcessError,
                json.JSONDecodeError,
            ):
                continue
            if (
                isinstance(record, dict)
                and isinstance(record.get("paper"), dict)
            ):
                aliases.update(
                    paper_aliases(record["paper"])
                )
    return aliases


def ingest_one(
    input_path: Path,
    repo_root: Path,
    config: dict[str, Any],
    external_aliases: set[str] | None = None,
    expected_collector: str | None = None,
) -> legacy.IngestResult:
    payload = legacy.read_json(input_path)
    validate_hardened_candidate(
        payload,
        repo_root,
        config,
        expected_collector,
    )
    payload = _sanitise_payload(payload)
    records = legacy.candidate_records(repo_root)

    match = _find_local_match(records, payload)
    if match is not None:
        output = (
            repo_root
            / "knowledge/literature/candidates"
            / f"{match['id']}.json"
        )
        changed = legacy.merge_record(match, payload)
        if changed:
            legacy.validate_record(match, output)
            legacy.write_json(output, match)
            return legacy.IngestResult(
                str(match["id"]),
                "merged",
                output,
                "added alias-matched independent discovery",
            )
        return legacy.IngestResult(
            str(match["id"]),
            "no-change",
            output,
            "schedule run already ingested",
        )

    incoming_aliases = paper_aliases(payload["paper"])
    if (
        external_aliases
        and incoming_aliases & external_aliases
    ):
        return legacy.IngestResult(
            canonical_paper_id(payload),
            "duplicate",
            None,
            "candidate alias already exists on another radar branch",
        )

    allowed, reason = quota_allows(
        records,
        payload,
        config,
    )
    if not allowed:
        return legacy.IngestResult(
            canonical_paper_id(payload),
            "quota-rejected",
            None,
            reason,
        )

    paper_id = canonical_paper_id(payload)
    output = (
        repo_root
        / "knowledge/literature/candidates"
        / f"{paper_id}.json"
    )
    record = legacy.new_record(payload, paper_id)
    legacy.validate_record(record, output)
    legacy.write_json(output, record)
    return legacy.IngestResult(
        paper_id,
        "created",
        output,
        "new canonical candidate",
    )


def update_daily_digest(
    repo_root: Path,
    date: str,
    config: dict[str, Any],
) -> Path:
    timezone = str(config.get("timezone", "Asia/Tokyo"))
    records = [
        record
        for record in legacy.candidate_records(repo_root)
        if record_date(record, timezone) == date
    ]
    auto = "\n".join(
        [
            legacy.AUTO_START,
            "## \u81ea\u52d5\u53ce\u96c6\u5019\u88dc",
            "",
            f"\u5019\u88dc\u6570: **{len(records)}**",
            "",
            *legacy._candidate_rows(records),
            legacy.AUTO_END,
        ]
    )
    path = (
        repo_root
        / "knowledge/literature/digests"
        / f"{date}.md"
    )
    preamble = SAFE_DIGEST_PREAMBLE.format(date=date)

    if path.exists():
        text = path.read_text(encoding="utf-8")
        start_count = text.count(legacy.AUTO_START)
        end_count = text.count(legacy.AUTO_END)
        if start_count != 1 or end_count != 1:
            raise legacy.CandidateError(
                f"{path}: digest markers must occur exactly once"
            )
        prefix, remainder = text.split(
            legacy.AUTO_START,
            1,
        )
        _, suffix = remainder.split(
            legacy.AUTO_END,
            1,
        )
        if (
            prefix.rstrip().endswith("`")
            or LEGACY_BROKEN_PREAMBLE in prefix
        ):
            prefix = preamble
        text = f"{prefix}{auto}{suffix}"
    else:
        text = (
            f"{preamble}{auto}\n\n"
            "## \u65e5\u6b21\u30ec\u30d3\u30e5\u30fc\n\n"
            "\u672a\u30ec\u30d3\u30e5\u30fc\u3002\n"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _quota_entry(
    accepted: int,
    limit: int,
) -> dict[str, int]:
    return {
        "accepted": accepted,
        "limit": limit,
        "remaining": max(0, limit - accepted),
    }


def update_daily_status(
    repo_root: Path,
    date: str,
    config: dict[str, Any],
) -> Path:
    timezone = str(config.get("timezone", "Asia/Tokyo"))
    records = legacy.candidate_records(repo_root)
    same_day = [
        record
        for record in records
        if record_date(record, timezone) == date
    ]
    settings = config["ingestion"]
    collector_limit = int(
        settings["max_candidates_per_collector_per_day"]
    )
    topic_limit = int(
        settings["max_candidates_per_topic_per_day"]
    )

    collectors = {
        collector_id: _quota_entry(
            sum(
                record_has_collector(
                    record,
                    collector_id,
                )
                for record in same_day
            ),
            collector_limit,
        )
        for collector_id in config["collectors"]
    }
    configured_topics = sorted(
        {
            topic
            for collector in config["collectors"].values()
            for topic in collector.get("topics", [])
        }
    )
    topics = {
        topic: _quota_entry(
            sum(
                topic in record_topics(record)
                for record in same_day
            ),
            topic_limit,
        )
        for topic in configured_topics
    }

    core = {
        "accepted_candidates": len(same_day),
        "daily_limit": int(
            settings["max_candidates_total_per_day"]
        ),
        "open_candidates": sum(
            record.get("state") == "inbox"
            for record in records
        ),
        "open_limit": int(settings["max_open_candidates"]),
        "collectors": collectors,
        "topics": topics,
    }
    path = (
        repo_root
        / "knowledge/literature/status"
        / f"{date}.json"
    )
    existing = (
        legacy.read_json(path)
        if path.exists()
        else {}
    )
    generated_at = existing.get("generated_at")
    if (
        existing.get("ingestion") != core
        or not isinstance(generated_at, str)
    ):
        generated_at = datetime.now(
            ZoneInfo(timezone)
        ).isoformat(timespec="seconds")

    status = {
        "schema_version": 1,
        "date": date,
        "initialized_at": existing.get("initialized_at"),
        "last_curated_at": existing.get(
            "last_curated_at"
        ),
        "generated_at": generated_at,
        "ingestion": core,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_json(path, status)
    return path


def validate_repository(repo_root: Path) -> list[str]:
    errors = legacy.validate_repository(repo_root)
    seen_aliases: dict[str, str] = {}

    for record in legacy.candidate_records(repo_root):
        paper_id = str(record["id"])
        for alias in paper_aliases(record["paper"]):
            previous = seen_aliases.get(alias)
            if previous is not None and previous != paper_id:
                errors.append(
                    "duplicate literature alias "
                    f"{alias}: {previous}, {paper_id}"
                )
            seen_aliases[alias] = paper_id

    digest_dir = (
        repo_root
        / "knowledge/literature/digests"
    )
    if digest_dir.exists():
        for path in sorted(digest_dir.glob("????-??-??.md")):
            text = path.read_text(encoding="utf-8")
            if (
                text.count(legacy.AUTO_START) != 1
                or text.count(legacy.AUTO_END) != 1
            ):
                errors.append(
                    f"{path}: digest markers must occur "
                    "exactly once"
                )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    date_parser = commands.add_parser("date")
    date_parser.add_argument("input", type=Path)
    date_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    date_parser.add_argument("--expected-collector")

    ingest_parser = commands.add_parser("ingest")
    ingest_parser.add_argument(
        "input",
        nargs="+",
        type=Path,
    )
    ingest_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    ingest_parser.add_argument("--dedup-ref-prefix")
    ingest_parser.add_argument("--expected-collector")
    ingest_parser.add_argument(
        "--update-digest",
        action="store_true",
    )
    ingest_parser.add_argument(
        "--update-status",
        action="store_true",
    )

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    try:
        config = legacy.load_config(repo_root)
        if args.command == "date":
            payload = legacy.read_json(args.input)
            validate_hardened_candidate(
                payload,
                repo_root,
                config,
                args.expected_collector,
            )
            print(legacy.local_date(payload, config))
            return 0

        if args.command == "validate":
            errors = validate_repository(repo_root)
            for error in errors:
                print(f"ERROR: {error}")
            print(
                f"{len(errors)} literature radar error(s)."
            )
            return 1 if errors else 0

        maximum = int(
            config["ingestion"][
                "max_candidates_per_hourly_run"
            ]
        )
        if len(args.input) > maximum:
            raise legacy.CandidateError(
                f"hourly run supplied {len(args.input)} "
                f"candidates; maximum is {maximum}"
            )

        payloads: list[tuple[Path, dict[str, Any]]] = []
        dates: set[str] = set()
        for input_path in args.input:
            payload = legacy.read_json(input_path)
            validate_hardened_candidate(
                payload,
                repo_root,
                config,
                args.expected_collector,
            )
            payloads.append((input_path, payload))
            dates.add(
                legacy.local_date(payload, config)
            )
        if len(dates) != 1:
            raise legacy.CandidateError(
                "one ingest invocation requires one "
                f"JST date: {sorted(dates)}"
            )

        ingest_date = next(iter(dates))
        external_aliases = aliases_in_git_refs(
            repo_root,
            args.dedup_ref_prefix,
            ingest_date,
            int(
                config["ingestion"][
                    "dedup_branch_window_days"
                ]
            ),
        )
        for input_path, _ in payloads:
            result = ingest_one(
                input_path,
                repo_root,
                config,
                external_aliases,
                args.expected_collector,
            )
            print(
                f"{result.action}: {result.paper_id} - "
                f"{result.message}"
            )

        if args.update_digest:
            print(
                "updated digest: "
                f"{update_daily_digest(repo_root, ingest_date, config)}"
            )
        if args.update_status:
            print(
                "updated status: "
                f"{update_daily_status(repo_root, ingest_date, config)}"
            )

        errors = validate_repository(repo_root)
        if errors:
            raise legacy.CandidateError(
                "; ".join(errors)
            )
        return 0
    except (
        legacy.CandidateError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
