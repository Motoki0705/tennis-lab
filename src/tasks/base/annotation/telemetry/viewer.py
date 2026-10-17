"""Offline index and per-response pages; images are loaded only when requested."""

from __future__ import annotations

import html
import os
from pathlib import Path

from ..artifacts import write_asset
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from .contracts import ResponseEntry, ResponseRecord


def _link(layout: AnnotationLayout, root: Path, ref: ArtifactRef) -> str:
    return html.escape(os.path.relpath(layout.artifact_path(ref), root), quote=True)


def page(title: str, content: str) -> bytes:
    return (
        f'<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        f"<title>{html.escape(title)}</title><style>body{{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem;line-height:1.6}}"
        "img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:.4rem;border:1px solid #ccc}pre{white-space:pre-wrap}</style>"
        f"<h1>{html.escape(title)}</h1>{content}</html>"
    ).encode()


def response_page(
    layout: AnnotationLayout, directory: Path, record: ResponseRecord
) -> ArtifactRef:
    path = directory / f"response-{record.index:05d}.html"
    usage = "usage未記録" if record.usage is None else record.usage.model_dump_json()
    parts = [
        f"<p>{html.escape(record.response_id)}</p><pre>{html.escape(usage)}</pre>",
        "<p>context近似は当該応答のinput + output。累積token数ではありません。</p>",
    ]
    for output in record.tool_outputs:
        parts.append(
            f"<h2>{html.escape(output.name or 'unknown tool')}</h2><p>{html.escape(output.category)} / "
            f"入力 {output.input_chars} chars / 出力文章 {output.text_chars} chars</p>"
            f"<p>{html.escape(output.input_timestamp or '開始時刻不明')} → {html.escape(output.timestamp or '終了時刻不明')}</p>"
        )
        for item in output.images:
            link = _link(layout, directory, item.image)
            metadata_link = (
                ""
                if item.source is None
                else f' / <a href="{_link(layout, directory, item.source.metadata)}">frame・crop・描画条件・注釈版</a>'
            )
            parts.append(
                f'<figure><a href="{link}"><img loading="lazy" src="{link}" alt="記録された入力画像"></a>'
                f"<figcaption>{item.width} × {item.height} / {html.escape(item.detail or 'unspecified')} / "
                f"{html.escape(item.binding)} / SHA256 {item.image.digest}{metadata_link}</figcaption></figure>"
            )
    return write_asset(layout, path, page(f"応答 {record.index}", "".join(parts)))


def index_page(
    layout: AnnotationLayout,
    directory: Path,
    entries: list[ResponseEntry],
    summary: str,
) -> ArtifactRef:
    rows = [
        "<table><tr><th>応答</th><th>入力</th><th>出力</th><th>新規画像</th><th>tool文章量</th></tr>"
    ]
    for entry in entries:
        rows.append(
            f'<tr><td><a href="{_link(layout, directory, entry.page)}">{entry.index}</a></td>'
            f"<td>{entry.input_tokens if entry.input_tokens is not None else '不明'}</td>"
            f"<td>{entry.output_tokens if entry.output_tokens is not None else '不明'}</td>"
            f"<td>{entry.images}</td><td>{entry.text_chars}</td></tr>"
        )
    rows.append("</table>")
    return write_asset(
        layout,
        directory / "index.html",
        page("注釈応答の記録", f"<p>{html.escape(summary)}</p>" + "".join(rows)),
    )
