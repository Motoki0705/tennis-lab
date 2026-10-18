"""Dedicated report observations. Shared five-layer memory is never written here."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from . import shell

DIRECTORY = ".agents/memory/ops-report"
SHARED_LAYERS = ("workflow", "decision", "environment", "gotcha", "review")
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
FIELDS = {
    "slug",
    "title",
    "focus",
    "observation",
    "evidence",
    "hypothesis",
    "deferred_reason",
    "recheck_when",
    "references",
    "last_verified",
}


def _text(value: Any, where: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(
            f"{where} must be a string{' (non-empty)' if not empty else ''}"
        )
    return value


def _date(value: Any, where: str) -> date:
    text = _text(value, where)
    parsed = date.fromisoformat(text)
    if parsed.isoformat() != text or parsed > date.today():
        raise ValueError(f"{where} must be an ISO date no later than today")
    return parsed


def _slug(value: Any) -> str:
    text = _text(value, "memory slug")
    if not SLUG.fullmatch(text) or text in {"index", "summary"}:
        raise ValueError(f"invalid memory slug: {text!r}")
    return text


def parse_update(raw: Any) -> dict[str, Any] | None:
    """A null is an explicit no-change; otherwise accept only structured patches.

    The writer owns filenames, frontmatter and index generation. Agents never
    return arbitrary paths or shell commands.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"summary", "upsert", "delete"}:
        raise ValueError(
            "memory_update requires exactly summary, upsert, delete (or null)"
        )
    _text(raw["summary"], "memory_update.summary")
    if not isinstance(raw["upsert"], list) or not isinstance(raw["delete"], list):
        raise ValueError("memory_update.upsert/delete must be lists")
    seen = set()
    for item in raw["upsert"]:
        if not isinstance(item, dict) or set(item) != FIELDS:
            raise ValueError(f"memory_update.upsert requires exactly {sorted(FIELDS)}")
        slug = _slug(item["slug"])
        if slug in seen:
            raise ValueError(f"duplicate memory slug: {slug}")
        seen.add(slug)
        for key in FIELDS - {"references", "last_verified", "slug"}:
            _text(
                item[key],
                f"memory_update.{key}",
                empty=key in {"hypothesis", "deferred_reason"},
            )
        if any(char in item["title"] for char in "\n\r[]"):
            raise ValueError("memory title must be one line without link brackets")
        if not isinstance(item["references"], list) or not item["references"]:
            raise ValueError("memory references must be a non-empty list")
        for ref in item["references"]:
            _text(ref, "memory reference")
        _date(item["last_verified"], "memory last_verified")
    for item in raw["delete"]:
        if not isinstance(item, dict) or set(item) != {"slug", "reason"}:
            raise ValueError("memory_update.delete requires slug and reason")
        slug = _slug(item["slug"])
        if slug in seen:
            raise ValueError(f"duplicate memory slug: {slug}")
        seen.add(slug)
        _text(item["reason"], "memory deletion reason")
    return raw


def read_tree(repo_root: Path, sha: str) -> dict[str, str]:
    """Read the committed version, even if the caller's checkout is dirty/old."""
    rows = shell.git(["ls-tree", "-r", "-z", sha, "--", DIRECTORY], cwd=repo_root)
    files = {}
    for row in rows.split("\0"):
        if not row:
            continue
        metadata, path = row.split("\t", 1)
        if metadata.split()[:2] != ["100644", "blob"]:
            raise ValueError(
                f"dedicated memory must contain regular Markdown files: {path}"
            )
        name = path.removeprefix(DIRECTORY + "/")
        if name not in {"SUMMARY.md", "index.md"}:
            _slug(name.removesuffix(".md"))
            if not name.endswith(".md"):
                raise ValueError(f"invalid dedicated memory file: {path}")
        files[name] = shell.git(["show", f"{sha}:{path}"], cwd=repo_root)
    if files and not {"SUMMARY.md", "index.md"} <= files.keys():
        raise ValueError("dedicated memory requires both SUMMARY.md and index.md")
    return files


def collect(repo_root: Path, checkout_base: str) -> dict[str, Any]:
    # A missing installation is reported explicitly, as for shared memory. Never
    # substitute checkout observations for the last valid origin/main version.
    ref = shell.git(
        ["show-ref", "--verify", "refs/remotes/origin/main"],
        cwd=repo_root,
        ok_codes=(0, 1, 128),
    )
    if not ref.strip():
        if read_tree(repo_root, checkout_base):
            raise ValueError(
                "dedicated memory requires origin/main; no checkout fallback"
            )
        return {
            "status": "not_installed",
            "note": "専用memory未導入（origin/mainなし）",
            "files": {},
        }
    # Imported here to keep collect.py's shared-layer collector independent.
    from .collect import pin_base

    base = pin_base(repo_root, "origin/main")
    files = read_tree(repo_root, base.sha)
    return {
        "status": "ok" if files else "not_installed",
        "note": "origin/mainの最後の有効版"
        if files
        else "専用memory未導入（origin/main）",
        "base_sha": base.sha,
        "files": files,
    }


def apply_update(
    files: dict[str, str], update: dict[str, Any] | None, *, source: str, today: date
) -> dict[str, str]:
    if update is None:
        return dict(files)
    parse_update(update)
    if not files:
        raise ValueError(
            "cannot update dedicated memory before it is installed on origin/main"
        )
    result = dict(files)
    for item in update["delete"]:
        name = item["slug"] + ".md"
        if name not in result:
            raise ValueError(f"cannot delete missing memory entry: {name}")
        del result[name]
    for item in update["upsert"]:
        name = item["slug"] + ".md"
        previous = _metadata(result[name]) if name in result else None
        verified = date.fromisoformat(item["last_verified"])
        created = previous["created"] if previous else verified
        if not created <= verified <= today or (
            previous and verified < previous["last_verified"]
        ):
            raise ValueError(f"memory dates must advance monotonically: {name}")
        fields = {
            "title": item["title"],
            "applies_to": "all",
            "source": source,
            "created": created,
            "last_verified": verified,
            "evidence": item["evidence"],
        }
        frontmatter = "\n".join(
            f"{key}: {value.isoformat() if isinstance(value, date) else json.dumps(value, ensure_ascii=False)}"
            for key, value in fields.items()
        )
        result[name] = (
            f"---\n{frontmatter}\n---\n\n"
            f"**注目領域:** {item['focus']}\n\n{item['observation']}\n\n"
            f"**未解決仮説:** {item['hypothesis'] or 'なし'}\n\n"
            f"**見送り理由:** {item['deferred_reason'] or 'なし'}\n\n"
            f"**次回の再検証条件:** {item['recheck_when']}\n\n"
            f"**参照:** {' / '.join(item['references'])}\n"
        )
    result["SUMMARY.md"] = (
        f"# 運用レポート専用memory\n\n{update['summary'].strip()}\n\n[観察の索引](index.md)\n"
    )
    result["index.md"] = (
        "# 観察の索引\n\n"
        + "".join(
            f"- [{_metadata(result[name])['title']}]({name})\n"
            for name in sorted(result)
            if name not in {"SUMMARY.md", "index.md"}
        )
    ).rstrip() + "\n"
    return result


def _metadata(text: str) -> dict[str, Any]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    data = yaml.safe_load(match[1]) if match else None
    if not isinstance(data, dict) or not isinstance(data.get("title"), str):
        raise ValueError("invalid dedicated memory frontmatter/title")
    for key in ("created", "last_verified"):
        if type(data.get(key)) is not date:
            raise ValueError(f"invalid dedicated memory {key}")
    return data


def load_decisions(path: Path | None) -> list[dict[str, Any]]:
    """Narrow inlet for #1223's triage; unchecked proposals are not rejections."""
    if path is None:
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("memory decisions must be a JSON list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "proposal_id",
            "decision",
            "reason",
            "source",
        }:
            raise ValueError(
                "memory decisions require proposal_id, decision, reason, source"
            )
        pid = row["proposal_id"]
        if (
            not isinstance(pid, str)
            or not re.fullmatch(r"R-\d{8}-\d{2,}", pid)
            or pid in seen
        ):
            raise ValueError(f"invalid/duplicate memory decision proposal_id: {pid!r}")
        seen.add(pid)
        if row["decision"] not in {"selected", "deferred"}:
            raise ValueError("memory decision must be selected or deferred")
        _text(row["reason"], "memory decision reason")
        _text(row["source"], "memory decision source")
    return rows
