#!/usr/bin/env python
"""Check the library's narrative documents: summary.md and its companions.

summary.md holds only the current judgement per task and must stay within a
character budget, so reading it before planning stays cheap. Relative links in
the maintained documents must resolve; frozen archives under ``history/`` are
checked only for links into the library itself. Whether a branch reviewed the
summary is a change rule in kg_changes.py (``--check-summary --base-ref``).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from kg_lib import nodes_dir
from kg_schema import has_prose

# Characters, not bytes. The 2026-11 rewrite is about half of it; raise it only deliberately.
SUMMARY_BUDGET = 20_000
LINK = re.compile(r"!?\[[^\]\n]*\]\(<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)
CODE = re.compile(r"`[^`\n]+`")


def library() -> Path:
    return nodes_dir().parent


def broken_links(path: Path, text: str, *, within: Path | None = None) -> list[str]:
    """Relative links (outside code) whose target does not exist."""
    errors = []
    prose = CODE.sub("", FENCE.sub("", text))
    for match in LINK.finditer(prose):
        parts = urlsplit(match.group(1))
        if parts.scheme or parts.netloc or not parts.path:
            continue
        target = (path.parent / unquote(parts.path)).resolve()
        if within is not None and not target.is_relative_to(within.resolve()):
            continue
        if not target.exists():
            errors.append(f"{path.relative_to(library())}: broken link {match.group(1)}")
    return errors


def check_summary() -> list[str]:
    base = library()
    path = base / "summary.md"
    if not path.is_file():
        return ["summary.md is missing"]
    text = path.read_text(encoding="utf-8")
    errors = []
    if not has_prose(text):
        errors.append("summary.md needs a substantive narrative, not just headings/comments")
    if len(text) > SUMMARY_BUDGET:
        errors.append(f"summary.md has {len(text)} characters (budget {SUMMARY_BUDGET}); keep only current judgements and move settled history into group nodes")
    maintained = [path, base / "README.md", base / "Papers/README.md", *sorted((base / "reports").glob("*.md"))]
    for document in maintained:
        if document.is_file():
            errors.extend(broken_links(document, document.read_text(encoding="utf-8")))
    for archive in sorted((base / "history").glob("*.md")):
        errors.extend(broken_links(archive, archive.read_text(encoding="utf-8"), within=base))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = check_summary()
    for error in errors:
        print(f"ERROR: {error}")
    path = library() / "summary.md"
    if path.is_file():
        print(f"summary.md: {len(path.read_text(encoding='utf-8'))}/{SUMMARY_BUDGET} characters")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
