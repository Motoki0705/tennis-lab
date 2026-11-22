"""Render the shared summary and indexes from typed entries without I/O."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .schema import (
    INDEX_NAME,
    LAYER_DESCRIPTIONS,
    MEMORY_DIR,
    SHARED_LAYERS,
    SUMMARY_PATH,
    Entry,
)

_SUMMARY_HEADER = """# 共有memory SUMMARY

各層の要点はエントリから自動生成する。作業を始める前に読み、詳細は層の索引から辿る。[memory skill](../skills/memory/SKILL.md) に読み方と更新手順がある。memoryの観察とコードが食い違った場合はコードを確認する。
"""
_INDEX_GUIDANCE = "エントリから自動生成する。更新手順は [memory skill](../../skills/memory/SKILL.md#形式) を参照。"


def render_documents(entries: Iterable[Entry]) -> dict[str, str]:
    """Return repo-relative paths and complete Markdown with LF line endings."""
    by_layer: dict[str, list[Entry]] = {layer: [] for layer in SHARED_LAYERS}
    for entry in entries:
        by_layer[entry.layer].append(entry)

    summary = [_SUMMARY_HEADER]
    documents: dict[str, str] = {}
    for layer, layer_entries in by_layer.items():
        description = LAYER_DESCRIPTIONS[layer]
        summary.append(f"\n## [{layer}]({layer}/{INDEX_NAME}) — {description}\n\n")
        index = [f"# {layer} — {description}\n\n{_INDEX_GUIDANCE}\n\n"]
        for entry in sorted(layer_entries, key=lambda entry: entry.filename):
            scope = f" ({entry.applies_to})" if entry.applies_to != "all" else ""
            label = re.sub(r"([\[\]\\])", r"\\\1", entry.title)
            index.append(f"- [{label}]({entry.filename}){scope}\n")
            summary.append(f"- {entry.summary}\n")
        documents[f"{MEMORY_DIR}/{layer}/{INDEX_NAME}"] = "".join(index)
    return {SUMMARY_PATH: "".join(summary), **documents}
