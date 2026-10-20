#!/usr/bin/env python
"""Plan or apply commit-map references as a normal commit after a history rewrite.

Only unique old commit prefixes (7-40 hex digits) are replaced, with full new
SHAs reachable from branches/tags/remotes. Foreign URLs, patches, deletion
provenance and content hashes remain intact. No history or remote is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any

from kg_artifacts import EXCEPTIONS, MANIFEST, TEXT, tree_sha256

TOKEN = re.compile(r"(?<![\w])[0-9a-fA-F]{7,40}(?![\w])")
FULL_SHA = re.compile(r"[0-9a-f]{40}\Z")
URL = re.compile(r"(?:https?://|git@)[^\s<>\]\)\"'`]+")
REFERENCE_PREFIX = re.compile(
    r"(?:\b[\w.-]*(?:commit|revision|git_head|head_sha|base_head)\b[\"']?\s*[:=：]?\s*"
    r"|\bSHA\s*[:=：]?\s*|コミット\s*[:=：]?\s*"
    r"|\bcheckout\s+"
    r"|\bworktree\s+add\s+--detach\s+\S+\s+"
    r"|\bgit\s+(?:checkout|show|switch|reset|rebase)\s+(?:--?[\w-]+\s+)*)[\"'`(]*$",
    re.I,
)
CONTENT_HASH_PREFIX = re.compile(
    r"\b(?:sha1|sha256|sha512|checksum|content_hash|blob_sha|blob_hash|tree_sha|tree_hash)"
    r"[\"']?\s*[:=]\s*[\"'`]*$",
    re.I,
)
LOCAL_URL = re.compile(
    r"https?://github\.com/Motoki0705/tennis-lab(?:\.git)?(?:/|$)", re.I
)
ZERO = "0" * 40
TEXT_SUFFIXES = TEXT | {
    ".rst",
    ".bash",
    ".zsh",
    ".cfg",
    ".ini",
    ".lock",
    ".xml",
    ".c",
    ".cpp",
    ".h",
    ".cu",
}


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def read_map(path: Path) -> dict[str, str]:
    rows = path.read_text().splitlines()
    if not rows or rows[0].split() != ["old", "new"]:
        raise ValueError("commit-map must start with 'old new'")
    mapping = {}
    for row in rows[1:]:
        old, new = row.split()
        if not FULL_SHA.fullmatch(old) or not FULL_SHA.fullmatch(new) or old in mapping:
            raise ValueError(f"invalid or duplicate commit-map row: {row}")
        mapping[old] = new
    if not mapping:
        raise ValueError("empty commit-map")
    return mapping


class Resolver:
    def __init__(self, mapping: dict[str, str], reachable: set[str]) -> None:
        self.mapping = mapping
        self.keys = sorted(mapping)
        self.new = {value for value in mapping.values() if value != ZERO}
        self.reachable = reachable
        self.cache: dict[str, tuple[str, str | None]] = {}

    def resolve(self, token: str) -> tuple[str, str | None]:
        token = token.lower()
        if token in self.cache:
            return self.cache[token]
        start = bisect_left(self.keys, token)
        stop = bisect_left(self.keys, token + "g")
        matches = self.keys[start:stop]
        result: tuple[str, str | None]
        if len(matches) > 1:
            result = ("ambiguous_old_prefix", None)
        elif not matches:
            result = ("already_new" if token in self.new else "unknown", None)
        else:
            old = matches[0]
            new = self.mapping[old]
            if new == ZERO:
                result = ("dropped_commit", None)
            elif new not in self.reachable:
                result = ("unpublished_target", new)
            elif new == old:
                result = ("unchanged_commit", None)
            elif token in self.new:
                result = ("ambiguous_old_and_new", None)
            else:
                result = ("replace", new)
        if result[0] != "unknown":
            self.cache[token] = result
        return result


def transform(text: str, resolver: Resolver) -> tuple[str, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    result = []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        foreign_urls = [m for m in URL.finditer(line) if not LOCAL_URL.match(m.group())]
        local_refs = [
            (m.start(), m.end())
            for m in URL.finditer(line)
            if LOCAL_URL.match(m.group())
            and re.search(r"/(?:commit|tree|blob|compare)/", m.group())
        ]
        foreign = [(m.start(), m.end()) for m in foreign_urls]
        foreign_ids = {
            t.lower() for m in foreign_urls for t in TOKEN.findall(m.group())
        }

        def replace(
            match: re.Match[str],
            foreign: list[tuple[int, int]] = foreign,
            line: str = line,
            number: int = number,
            foreign_ids: set[str] = foreign_ids,
            local_refs: list[tuple[int, int]] = local_refs,
        ) -> str:
            token = match.group()
            if any(start <= match.start() < end for start, end in foreign) or any(
                oid.startswith(token.lower()) for oid in foreign_ids
            ):
                return token
            prefix = line[max(0, match.start() - 100) : match.start()]
            if CONTENT_HASH_PREFIX.search(prefix):
                return token
            explicit = (
                bool(REFERENCE_PREFIX.search(prefix))
                or (
                    match.start() > 0
                    and line[match.start() - 1] == "`"
                    and line[match.end() : match.end() + 1] == "`"
                    and not token.isdigit()
                )
                or any(start <= match.start() < end for start, end in local_refs)
                or bool(re.match(r"\s*\(the commit\b", line[match.end() :]))
            )
            # Large numeric logs/figures must not be interpreted as short SHAs.
            if len(token) < 40 and not explicit:
                return token
            status, new = resolver.resolve(token)
            # Bare unknown hex strings can be hashes, IDs, or colors. Record
            # them separately rather than asserting they are missing commits.
            if status == "unknown":
                status = "unknown_commit_reference" if explicit else "unclassified_hex"
            if status not in {"already_new", "unchanged_commit", "unclassified_hex"}:
                events.append(dict(line=number, old=token, new=new, status=status))
            return new if status == "replace" and new is not None else token

        result.append(TOKEN.sub(replace, line))
    return "".join(result), events


def plan(
    root: Path,
    mapping: dict[str, str],
    *,
    text_dir: Path | None = None,
    refresh_run_exceptions: bool = False,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    # refs/pull and replace refs are intentionally not publication refs.
    reachable = set(
        git(root, "rev-list", "--branches", "--tags", "--remotes").decode().splitlines()
    )
    resolver = Resolver(mapping, reachable)
    changes: dict[Path, bytes] = {}
    events: list[dict[str, Any]] = []
    protected = []
    baseline: dict[Path, bytes] = {}
    if text_dir is None:
        files = []
        for row in git(root, "ls-files", "--stage", "-z").split(b"\0"):
            if not row:
                continue
            info, name = row.split(b"\t", 1)
            mode, _, stage = info.split()
            if stage != b"0":
                raise ValueError("unmerged index; resolve it before planning")
            if mode in {b"100644", b"100755"}:
                files.append(root / name.decode())
        policy = root / "knowledge" / EXCEPTIONS
        exceptions = json.loads(policy.read_text()) if policy.exists() else None
        if exceptions:
            # Verify pinned run contents before allowing a narrow rebaseline.
            # The exception set and file set cannot grow through this operation.
            for row in exceptions["runs"]:
                directory = root / "knowledge" / row["path"]
                if tree_sha256(directory) != row["sha256"]:
                    raise ValueError(
                        f"pinned run differs before remapping: {row['path']}"
                    )
    else:
        files = sorted(text_dir.rglob("*.md"))
        exceptions = None
        policy = None
    for path in files:
        relative = str(path.relative_to(root if text_dir is None else text_dir))
        if path.is_symlink():
            raise ValueError(f"tracked regular file became a symlink: {relative}")
        if not path.resolve().is_relative_to(
            root if text_dir is None else text_dir.resolve()
        ):
            raise ValueError(
                f"input path escapes the checkout/text directory: {relative}"
            )
        data = path.read_bytes()
        baseline[path] = data
        if path.suffix in {".patch", ".diff"} or path.name in {MANIFEST, EXCEPTIONS}:
            protected.append(relative)
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "README",
            "LICENSE",
            "NOTICE",
        }:
            continue
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        updated, occurrences = transform(text, resolver)
        events.extend(dict(item, path=relative) for item in occurrences)
        if updated != text:
            changes[path] = updated.encode("utf-8")
    rebaselines = []
    if exceptions and policy is not None:
        for row in exceptions["runs"]:
            directory = root / "knowledge" / row["path"]
            affected = [p for p in changes if p.is_relative_to(directory)]
            if not affected:
                continue
            if not refresh_run_exceptions:
                raise ValueError(
                    "pinned run needs --refresh-run-exceptions (same file set, commit references only)"
                )
            lines = []
            for path in sorted(p for p in directory.rglob("*") if p.is_file()):
                data = changes.get(path, path.read_bytes())
                lines.append(
                    f"{hashlib.sha256(data).hexdigest()}  {path.relative_to(directory).as_posix()}\n"
                )
            new = hashlib.sha256("".join(lines).encode()).hexdigest()
            rebaselines.append(
                dict(
                    path=row["path"],
                    old=row["sha256"],
                    new=new,
                    changed_files=[str(p.relative_to(directory)) for p in affected],
                )
            )
            row["sha256"] = new
        if rebaselines:
            changes[policy] = (
                json.dumps(exceptions, ensure_ascii=False, indent=2) + "\n"
            ).encode()
    counts = dict(Counter(item["status"] for item in events))
    report = dict(
        counts=counts,
        files_changed=len(changes),
        occurrences=events,
        protected_files=protected,
        run_exception_updates=rebaselines,
        before_sha256={
            str(p.relative_to(root if text_dir is None else text_dir)): hashlib.sha256(
                baseline.get(p, p.read_bytes())
            ).hexdigest()
            for p in changes
        },
        after_sha256={
            str(p.relative_to(root if text_dir is None else text_dir)): hashlib.sha256(
                data
            ).hexdigest()
            for p, data in changes.items()
        },
    )
    return changes, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--commit-map", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--text-dir",
        type=Path,
        help="offline issue/card Markdown copies; never sends updates",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="apply known mappings while explicitly leaving reported unknown/unpublished commits",
    )
    parser.add_argument(
        "--refresh-run-exceptions",
        action="store_true",
        help="verify old hashes and rehash affected fixed runs; never adds paths or changes patch/deletion evidence",
    )
    args = parser.parse_args()
    root = args.repo.resolve()
    if args.text_dir is not None:
        args.text_dir = args.text_dir.resolve()
    if args.write and args.text_dir is None:
        if git(root, "status", "--porcelain=v1", "--untracked-files=no").strip():
            raise ValueError(
                "--write requires a clean tracked checkout; review existing changes first"
            )
    mapping = read_map(args.commit_map)
    changes, report = plan(
        root,
        mapping,
        text_dir=args.text_dir,
        refresh_run_exceptions=args.refresh_run_exceptions,
    )
    counts = report["counts"]
    ambiguous = sum(
        counts.get(key, 0)
        for key in ("ambiguous_old_prefix", "ambiguous_old_and_new", "dropped_commit")
    )
    unresolved = sum(
        counts.get(key, 0) for key in ("unknown_commit_reference", "unpublished_target")
    )
    blocked = bool(ambiguous or (unresolved and not args.allow_unresolved))
    report.update(
        commit_map_sha256=hashlib.sha256(args.commit_map.read_bytes()).hexdigest(),
        head=git(root, "rev-parse", "HEAD").decode().strip(),
        applied=False,
        blocked=blocked,
    )
    # Report output must not overwrite any tracked file or source text.
    report_path = args.report.resolve()
    if report_path.is_relative_to(root) or (
        args.text_dir and report_path.is_relative_to(args.text_dir.resolve())
    ):
        raise ValueError("write the report outside the checkout/text input directory")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.write and not blocked:
        base = root if args.text_dir is None else args.text_dir.resolve()
        for path in changes:
            name = str(path.relative_to(base))
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(base)
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != report["before_sha256"][name]
            ):
                raise ValueError(
                    f"file changed while planning; no writes applied: {name}"
                )
        for path, data in changes.items():
            path.write_bytes(data)
        report["applied"] = True
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                counts=counts,
                files_changed=len(changes),
                applied=report["applied"],
                blocked=blocked,
            )
        )
    )
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
