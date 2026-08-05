#!/usr/bin/env python
"""Validate and ingest scheduled literature discoveries.

The hourly ChatGPT schedules only write untrusted JSON files to dedicated queue
branches.  This script is the trusted GitHub Actions boundary: it validates the
payload, computes the canonical paper id, deduplicates it, enforces quotas, and
writes a canonical candidate record to the active daily branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

SCHEMA_VERSION = 1
KIND = "literature_candidate"
CANDIDATE_STATES = {"inbox", "reviewed", "rejected", "promoted"}
AUTO_START = "<!-- literature-radar:auto:start -->"
AUTO_END = "<!-- literature-radar:auto:end -->"


class CandidateError(ValueError):
    """Raised when a raw or canonical literature candidate is invalid."""


@dataclass(frozen=True)
class IngestResult:
    paper_id: str
    action: str
    path: Path | None
    message: str


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateError(f"{path}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateError(f"{path}: top-level JSON value must be an object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_config(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "knowledge/literature/config.json"
    config = read_json(path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise CandidateError(
            f"{path}: schema_version must be {SCHEMA_VERSION}, "
            f"got {config.get('schema_version')!r}"
        )
    return config


def parse_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise CandidateError(f"{field} must be a non-empty ISO-8601 string")
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CandidateError(f"{field} is not valid ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise CandidateError(f"{field} must include an explicit UTC offset")
    return parsed


def local_date(payload: dict[str, Any], config: dict[str, Any]) -> str:
    timezone = ZoneInfo(str(config.get("timezone", "Asia/Tokyo")))
    return parse_datetime(payload.get("discovered_at"), "discovered_at").astimezone(
        timezone
    ).date().isoformat()


def _require_text(mapping: dict[str, Any], key: str, prefix: str = "") -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        name = f"{prefix}.{key}" if prefix else key
        raise CandidateError(f"{name} must be a non-empty string")
    return value.strip()


def _require_score(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
        raise CandidateError(f"screening.{key} must be an integer from 0 to 100")
    return value


def _safe_repo_path(repo_root: Path, raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise CandidateError("screening.repo_paths entries must be non-empty strings")
    relative = Path(raw.strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise CandidateError(f"repository path must be relative and safe: {raw!r}")
    if not (repo_root / relative).exists():
        raise CandidateError(f"repository path does not exist: {raw!r}")
    return relative.as_posix()


def _http_url(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.startswith(("https://", "http://")):
        raise CandidateError(f"{field} must be an http(s) URL")
    return value


def validate_raw_candidate(
    payload: dict[str, Any],
    repo_root: Path,
    config: dict[str, Any],
    expected_collector: str | None = None,
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CandidateError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("kind") != KIND:
        raise CandidateError(f"kind must be {KIND!r}")

    collectors = config.get("collectors")
    collector_id = _require_text(payload, "collector_id")
    if not isinstance(collectors, dict) or collector_id not in collectors:
        raise CandidateError(f"collector_id is not configured: {collector_id!r}")
    if expected_collector is not None and collector_id != expected_collector:
        raise CandidateError(
            f"collector_id {collector_id!r} does not match queue branch "
            f"collector {expected_collector!r}"
        )
    _require_text(payload, "schedule_run_id")
    parse_datetime(payload.get("discovered_at"), "discovered_at")

    paper = payload.get("paper")
    if not isinstance(paper, dict):
        raise CandidateError("paper must be an object")
    _require_text(paper, "title", "paper")
    authors = paper.get("authors")
    if not isinstance(authors, list) or not authors or not all(
        isinstance(author, str) and author.strip() for author in authors
    ):
        raise CandidateError("paper.authors must be a non-empty string list")
    year = paper.get("year")
    if not isinstance(year, int) or isinstance(year, bool) or not 1900 <= year <= 2100:
        raise CandidateError("paper.year must be an integer from 1900 to 2100")

    identifiers = paper.get("identifiers")
    if not isinstance(identifiers, dict):
        raise CandidateError("paper.identifiers must be an object")
    for key in ("doi", "arxiv", "openreview"):
        value = identifiers.get(key)
        if value is not None and not isinstance(value, str):
            raise CandidateError(f"paper.identifiers.{key} must be a string or null")

    urls = paper.get("urls")
    if not isinstance(urls, dict):
        raise CandidateError("paper.urls must be an object")
    _http_url(urls.get("primary"), "paper.urls.primary")
    for key, value in urls.items():
        if value is not None:
            _http_url(value, f"paper.urls.{key}")

    screening = payload.get("screening")
    if not isinstance(screening, dict):
        raise CandidateError("screening must be an object")
    tasks = screening.get("tasks")
    allowed_tasks = set(config.get("allowed_tasks") or [])
    if not isinstance(tasks, list) or not tasks:
        raise CandidateError("screening.tasks must be a non-empty list")
    unknown_tasks = {str(task) for task in tasks} - allowed_tasks
    if unknown_tasks:
        raise CandidateError(f"screening.tasks contains unknown tasks: {sorted(unknown_tasks)}")

    collector_tasks = set(collectors[collector_id].get("tasks") or [])
    if collector_tasks and not set(str(task) for task in tasks).intersection(collector_tasks):
        raise CandidateError(
            f"candidate tasks do not overlap collector {collector_id!r} responsibility"
        )

    repo_paths = screening.get("repo_paths")
    if not isinstance(repo_paths, list) or not repo_paths:
        raise CandidateError("screening.repo_paths must be a non-empty list")
    for repo_path in repo_paths:
        _safe_repo_path(repo_root, repo_path)

    relevance = _require_score(screening, "relevance_score")
    _require_score(screening, "novelty_score")
    minimum = int(config.get("ingestion", {}).get("minimum_relevance_score", 0))
    if relevance < minimum:
        raise CandidateError(
            f"screening.relevance_score {relevance} is below configured minimum {minimum}"
        )

    evidence_level = _require_text(screening, "evidence_level", "screening")
    allowed_levels = set(config.get("evidence_levels") or [])
    if evidence_level not in allowed_levels:
        raise CandidateError(
            f"screening.evidence_level must be one of {sorted(allowed_levels)}"
        )
    for key in ("summary_ja", "applicability_ja", "risks_ja", "candidate_experiment_ja"):
        _require_text(screening, key, "screening")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CandidateError("sources must be a non-empty list")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise CandidateError(f"sources[{index}] must be an object")
        _require_text(source, "kind", f"sources[{index}]")
        _http_url(source.get("url"), f"sources[{index}].url")
        parse_datetime(source.get("checked_at"), f"sources[{index}].checked_at")


def _slug(value: str, max_length: int = 96) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{slug[: max_length - 13].rstrip('-')}-{digest}"


def _normalise_doi(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text.removeprefix("doi:").strip()


def _normalise_arxiv(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", text)
    text = text.removesuffix(".pdf")
    return re.sub(r"v\d+$", "", text)


def _normalise_openreview(value: str) -> str:
    text = value.strip()
    match = re.search(r"[?&]id=([^&]+)", text)
    return match.group(1) if match else text


def canonical_paper_id(payload: dict[str, Any]) -> str:
    paper = payload["paper"]
    identifiers = paper.get("identifiers") or {}
    doi = identifiers.get("doi")
    if isinstance(doi, str) and doi.strip():
        return f"paper-doi-{_slug(_normalise_doi(doi))}"
    arxiv = identifiers.get("arxiv")
    if isinstance(arxiv, str) and arxiv.strip():
        return f"paper-arxiv-{_slug(_normalise_arxiv(arxiv))}"
    openreview = identifiers.get("openreview")
    if isinstance(openreview, str) and openreview.strip():
        return f"paper-openreview-{_slug(_normalise_openreview(openreview))}"

    title = str(paper["title"]).strip().lower()
    year = int(paper["year"])
    digest = hashlib.sha256(f"{title}\n{year}".encode("utf-8")).hexdigest()[:16]
    return f"paper-title-{year}-{digest}"


def _clean_identifiers(raw: dict[str, Any]) -> dict[str, str | None]:
    output: dict[str, str | None] = {"doi": None, "arxiv": None, "openreview": None}
    for key in output:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            if key == "doi":
                output[key] = _normalise_doi(value)
            elif key == "arxiv":
                output[key] = _normalise_arxiv(value)
            else:
                output[key] = _normalise_openreview(value)
    return output


def _discovery(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "collector_id": payload["collector_id"],
        "schedule_run_id": payload["schedule_run_id"],
        "discovered_at": payload["discovered_at"],
        "screening": payload["screening"],
        "sources": payload["sources"],
    }


def _aggregate(discoveries: list[dict[str, Any]]) -> dict[str, Any]:
    relevance = [int(item["screening"]["relevance_score"]) for item in discoveries]
    novelty = [int(item["screening"]["novelty_score"]) for item in discoveries]
    tasks = sorted(
        {
            str(task)
            for item in discoveries
            for task in item["screening"].get("tasks", [])
        }
    )
    repo_paths = sorted(
        {
            str(path)
            for item in discoveries
            for path in item["screening"].get("repo_paths", [])
        }
    )
    collectors = sorted({str(item["collector_id"]) for item in discoveries})
    return {
        "collector_count": len(collectors),
        "collectors": collectors,
        "discovery_count": len(discoveries),
        "max_relevance_score": max(relevance),
        "mean_relevance_score": round(sum(relevance) / len(relevance), 2),
        "max_novelty_score": max(novelty),
        "tasks": tasks,
        "repo_paths": repo_paths,
    }


def new_record(payload: dict[str, Any], paper_id: str) -> dict[str, Any]:
    discovery = _discovery(payload)
    paper = payload["paper"]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": paper_id,
        "state": "inbox",
        "first_seen": payload["discovered_at"],
        "last_seen": payload["discovered_at"],
        "paper": {
            "title": paper["title"],
            "authors": list(paper["authors"]),
            "year": paper["year"],
            "venue": paper.get("venue"),
            "identifiers": _clean_identifiers(paper.get("identifiers") or {}),
            "urls": paper["urls"],
        },
        "aggregate": _aggregate([discovery]),
        "discoveries": [discovery],
        "curation": {
            "reviewed_at": None,
            "decision": None,
            "reason_ja": None,
            "paper_node": None,
            "proposal_nodes": [],
            "issue": None,
        },
    }


def merge_record(record: dict[str, Any], payload: dict[str, Any]) -> bool:
    discovery = _discovery(payload)
    discoveries = record.get("discoveries")
    if not isinstance(discoveries, list):
        raise CandidateError(f"{record.get('id')}: discoveries must be a list")
    key = (discovery["collector_id"], discovery["schedule_run_id"])
    existing_keys = {
        (item.get("collector_id"), item.get("schedule_run_id"))
        for item in discoveries
        if isinstance(item, dict)
    }
    if key in existing_keys:
        return False

    discoveries.append(discovery)
    discoveries.sort(key=lambda item: str(item.get("discovered_at", "")))
    record["last_seen"] = max(str(item["discovered_at"]) for item in discoveries)
    record["aggregate"] = _aggregate(discoveries)

    paper = record.setdefault("paper", {})
    incoming_paper = payload["paper"]
    authors = sorted(
        {
            str(author)
            for author in list(paper.get("authors") or [])
            + list(incoming_paper.get("authors") or [])
        }
    )
    paper["authors"] = authors
    identifiers = paper.setdefault("identifiers", {})
    for key_name, value in _clean_identifiers(
        incoming_paper.get("identifiers") or {}
    ).items():
        if value and not identifiers.get(key_name):
            identifiers[key_name] = value
    urls = paper.setdefault("urls", {})
    for key_name, value in (incoming_paper.get("urls") or {}).items():
        if value and not urls.get(key_name):
            urls[key_name] = value
    return True


def candidate_records(repo_root: Path) -> list[dict[str, Any]]:
    directory = repo_root / "knowledge/literature/candidates"
    records: list[dict[str, Any]] = []
    if not directory.exists():
        return records
    for path in sorted(directory.glob("paper-*.json")):
        record = read_json(path)
        validate_record(record, path)
        records.append(record)
    return records


def validate_record(record: dict[str, Any], path: Path | None = None) -> None:
    label = str(path) if path else str(record.get("id", "candidate"))
    if record.get("schema_version") != SCHEMA_VERSION:
        raise CandidateError(f"{label}: unsupported schema_version")
    paper_id = record.get("id")
    if not isinstance(paper_id, str) or not paper_id.startswith("paper-"):
        raise CandidateError(f"{label}: id must start with 'paper-'")
    if record.get("state") not in CANDIDATE_STATES:
        raise CandidateError(f"{label}: invalid state {record.get('state')!r}")
    first_seen = parse_datetime(record.get("first_seen"), f"{label}.first_seen")
    last_seen = parse_datetime(record.get("last_seen"), f"{label}.last_seen")
    if last_seen < first_seen:
        raise CandidateError(f"{label}: last_seen precedes first_seen")
    paper = record.get("paper")
    if not isinstance(paper, dict):
        raise CandidateError(f"{label}: paper must be an object")
    _require_text(paper, "title", f"{label}.paper")
    authors = paper.get("authors")
    if not isinstance(authors, list) or not authors:
        raise CandidateError(f"{label}: paper.authors must be non-empty")
    identifiers = paper.get("identifiers")
    if not isinstance(identifiers, dict):
        raise CandidateError(f"{label}: paper.identifiers must be an object")
    urls = paper.get("urls")
    if not isinstance(urls, dict):
        raise CandidateError(f"{label}: paper.urls must be an object")
    _http_url(urls.get("primary"), f"{label}.paper.urls.primary")
    discoveries = record.get("discoveries")
    if not isinstance(discoveries, list) or not discoveries:
        raise CandidateError(f"{label}: discoveries must be non-empty")
    aggregate = record.get("aggregate")
    if not isinstance(aggregate, dict):
        raise CandidateError(f"{label}: aggregate must be an object")
    if aggregate != _aggregate(discoveries):
        raise CandidateError(f"{label}: aggregate does not match discoveries")
    curation = record.get("curation")
    if not isinstance(curation, dict):
        raise CandidateError(f"{label}: curation must be an object")
    decision = curation.get("decision")
    if decision not in {None, "reviewed", "rejected", "promoted"}:
        raise CandidateError(f"{label}: invalid curation.decision {decision!r}")
    state = record.get("state")
    if state != "inbox" and decision != state:
        raise CandidateError(
            f"{label}: state {state!r} must match curation.decision {decision!r}"
        )
    paper_node = curation.get("paper_node")
    if state == "promoted" and (
        not isinstance(paper_node, str) or paper_node != paper_id
    ):
        raise CandidateError(
            f"{label}: promoted candidate requires curation.paper_node == id"
        )
    proposal_nodes = curation.get("proposal_nodes")
    if not isinstance(proposal_nodes, list) or not all(
        isinstance(item, str) for item in proposal_nodes
    ):
        raise CandidateError(f"{label}: curation.proposal_nodes must be a string list")
    issue = curation.get("issue")
    if issue is not None and (not isinstance(issue, int) or isinstance(issue, bool)):
        raise CandidateError(f"{label}: curation.issue must be an integer or null")


def _date_of_iso(value: str, timezone: str) -> str:
    return parse_datetime(value, "timestamp").astimezone(ZoneInfo(timezone)).date().isoformat()


def _quota_allows(
    records: list[dict[str, Any]],
    payload: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, str]:
    settings = config.get("ingestion", {})
    timezone = str(config.get("timezone", "Asia/Tokyo"))
    date = local_date(payload, config)
    collector_id = str(payload["collector_id"])
    same_day = [
        record
        for record in records
        if _date_of_iso(str(record["first_seen"]), timezone) == date
    ]
    collector_count = sum(
        1
        for record in same_day
        if any(
            discovery.get("collector_id") == collector_id
            for discovery in record.get("discoveries", [])
            if isinstance(discovery, dict)
        )
    )
    per_collector = int(settings.get("max_candidates_per_collector_per_day", 4))
    if collector_count >= per_collector:
        return False, f"collector daily quota reached ({collector_count}/{per_collector})"

    daily_total = int(settings.get("max_candidates_total_per_day", 12))
    if len(same_day) >= daily_total:
        return False, f"global daily quota reached ({len(same_day)}/{daily_total})"

    open_limit = int(settings.get("max_open_candidates", 60))
    open_count = sum(1 for record in records if record.get("state") == "inbox")
    if open_count >= open_limit:
        return False, f"open candidate backlog limit reached ({open_count}/{open_limit})"
    return True, ""


def candidate_ids_in_git_refs(
    repo_root: Path,
    ref_prefix: str | None,
    target_date: str | None = None,
    window_days: int = 14,
) -> set[str]:
    if not ref_prefix:
        return set()
    refs_output = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)", ref_prefix],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    ids: set[str] = set()
    target = (
        datetime.strptime(target_date, "%Y-%m-%d").date() if target_date else None
    )
    for ref in (line.strip() for line in refs_output.splitlines() if line.strip()):
        if target is not None:
            branch_name = ref.rsplit("/", 1)[-1]
            try:
                branch_date = datetime.strptime(branch_name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if abs((branch_date - target).days) > window_days:
                continue
        paths_output = subprocess.run(
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
        for path in (line.strip() for line in paths_output.splitlines() if line.strip()):
            if not path.endswith(".json") or not Path(path).name.startswith("paper-"):
                continue
            content = subprocess.run(
                ["git", "show", f"{ref}:{path}"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            try:
                record = json.loads(content)
            except json.JSONDecodeError:
                continue
            paper_id = record.get("id") if isinstance(record, dict) else None
            if isinstance(paper_id, str):
                ids.add(paper_id)
    return ids


def ingest_one(
    input_path: Path,
    repo_root: Path,
    config: dict[str, Any],
    external_ids: set[str] | None = None,
    expected_collector: str | None = None,
) -> IngestResult:
    payload = read_json(input_path)
    validate_raw_candidate(payload, repo_root, config, expected_collector)
    paper_id = canonical_paper_id(payload)
    output = repo_root / "knowledge/literature/candidates" / f"{paper_id}.json"

    if output.exists():
        record = read_json(output)
        validate_record(record, output)
        changed = merge_record(record, payload)
        if changed:
            validate_record(record, output)
            write_json(output, record)
            return IngestResult(paper_id, "merged", output, "added independent discovery")
        return IngestResult(paper_id, "no-change", output, "schedule run already ingested")

    if external_ids and paper_id in external_ids:
        return IngestResult(
            paper_id,
            "duplicate",
            None,
            "candidate already exists on another daily radar branch",
        )

    records = candidate_records(repo_root)
    allowed, reason = _quota_allows(records, payload, config)
    if not allowed:
        return IngestResult(paper_id, "quota-rejected", None, reason)

    record = new_record(payload, paper_id)
    validate_record(record, output)
    write_json(output, record)
    return IngestResult(paper_id, "created", output, "new canonical candidate")


def _candidate_rows(records: Iterable[dict[str, Any]]) -> list[str]:
    rows = [
        "| ID | 状態 | 関連度 | 収集数 | タスク | 論文 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for record in sorted(
        records,
        key=lambda item: (
            -int(item["aggregate"]["max_relevance_score"]),
            str(item["id"]),
        ),
    ):
        paper = record["paper"]
        aggregate = record["aggregate"]
        primary = paper.get("urls", {}).get("primary", "")
        title = str(paper.get("title", "")).replace("|", "\\|")
        paper_link = f"[{title}]({primary})" if primary else title
        rows.append(
            "| `{id}` | {state} | {score} | {count} | {tasks} | {paper} |".format(
                id=record["id"],
                state=record["state"],
                score=aggregate["max_relevance_score"],
                count=aggregate["discovery_count"],
                tasks=", ".join(aggregate["tasks"]),
                paper=paper_link,
            )
        )
    return rows


def update_daily_digest(repo_root: Path, date: str, config: dict[str, Any]) -> Path:
    timezone = str(config.get("timezone", "Asia/Tokyo"))
    records = [
        record
        for record in candidate_records(repo_root)
        if _date_of_iso(str(record["first_seen"]), timezone) == date
    ]
    auto = "\n".join(
        [
            AUTO_START,
            "## 自動収集候補",
            "",
            f"候補数: **{len(records)}**",
            "",
            *_candidate_rows(records),
            AUTO_END,
        ]
    )
    path = repo_root / "knowledge/literature/digests" / f"{date}.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if AUTO_START in text and AUTO_END in text:
            prefix, remainder = text.split(AUTO_START, 1)
            _, suffix = remainder.split(AUTO_END, 1)
            text = f"{prefix}{auto}{suffix}"
        else:
            text = f"{text.rstrip()}\n\n{auto}\n"
    else:
        text = (
            f"# Literature Radar — {date}\n\n"
            f"この日次ダイジェストの `{AUTO_START}` 区間はGitHub Actionsが更新します。\n"
            "日次curatorは自動区間の外側へレビュー結果を追記してください。\n\n"
            f"{auto}\n\n"
            "## 日次レビュー\n\n"
            "未レビュー。\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def validate_repository(repo_root: Path) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for path in sorted((repo_root / "knowledge/literature/candidates").glob("paper-*.json")):
        try:
            record = read_json(path)
            validate_record(record, path)
            paper_id = str(record["id"])
            if paper_id in seen:
                errors.append(f"{path}: duplicate candidate id {paper_id}")
            seen.add(paper_id)
            if path.stem != paper_id:
                errors.append(f"{path}: filename must match id {paper_id}")
        except CandidateError as exc:
            errors.append(str(exc))
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    date_parser = subparsers.add_parser("date", help="print the JST ingest date")
    date_parser.add_argument("input", type=Path)
    date_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    date_parser.add_argument("--expected-collector")

    ingest_parser = subparsers.add_parser("ingest", help="ingest one or more raw files")
    ingest_parser.add_argument("input", nargs="+", type=Path)
    ingest_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    ingest_parser.add_argument(
        "--dedup-ref-prefix",
        help="git ref prefix containing other open daily radar branches",
    )
    ingest_parser.add_argument("--update-digest", action="store_true")
    ingest_parser.add_argument("--expected-collector")

    validate_parser = subparsers.add_parser("validate", help="validate canonical records")
    validate_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    try:
        config = load_config(repo_root)
        if args.command == "date":
            payload = read_json(args.input)
            validate_raw_candidate(
                payload, repo_root, config, args.expected_collector
            )
            print(local_date(payload, config))
            return 0
        if args.command == "validate":
            errors = validate_repository(repo_root)
            for error in errors:
                print(f"ERROR: {error}")
            print(f"{len(errors)} literature candidate error(s).")
            return 1 if errors else 0

        maximum_inputs = int(
            config.get("ingestion", {}).get("max_candidates_per_hourly_run", 1)
        )
        if len(args.input) > maximum_inputs:
            raise CandidateError(
                f"hourly run supplied {len(args.input)} candidates; maximum is "
                f"{maximum_inputs}"
            )
        payloads: list[tuple[Path, dict[str, Any]]] = []
        dates: set[str] = set()
        for input_path in args.input:
            payload = read_json(input_path)
            validate_raw_candidate(
                payload, repo_root, config, args.expected_collector
            )
            payloads.append((input_path, payload))
            dates.add(local_date(payload, config))
        if len(dates) != 1:
            raise CandidateError(
                f"one ingest invocation must contain exactly one local date, got {sorted(dates)}"
            )
        ingest_date = next(iter(dates))
        window_days = int(
            config.get("ingestion", {}).get("dedup_branch_window_days", 14)
        )
        external_ids = candidate_ids_in_git_refs(
            repo_root,
            args.dedup_ref_prefix,
            ingest_date,
            window_days,
        )
        for input_path, _payload in payloads:
            result = ingest_one(
                input_path,
                repo_root,
                config,
                external_ids,
                args.expected_collector,
            )
            print(f"{result.action}: {result.paper_id} — {result.message}")
        if args.update_digest:
            path = update_daily_digest(repo_root, ingest_date, config)
            print(f"updated digest: {path}")
        errors = validate_repository(repo_root)
        if errors:
            raise CandidateError("; ".join(errors))
        return 0
    except (CandidateError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
