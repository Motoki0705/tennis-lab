"""Presentation-only bridge; stored/verified documents and exports stay byte exact.

The response retains the content endpoint's opaque-origin sandbox and renderer
hash policy. Only our fixed bridge's hash is added, never generated script bytes.
The HTTP policy replaces document CSP meta elements in this ephemeral view so a
visual compiler's script-src does not disable the trusted selection bridge.
"""

from __future__ import annotations

import base64
import hashlib
import re
from html import escape
from html.parser import HTMLParser
from pathlib import Path

from .schema import ContractError, Revision


class _Markup(HTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=False)
        self.offsets = [0, *(match.end() for match in re.finditer("\n", html))]
        self.policies: list[tuple[int, int]] = []
        self.body_end: int | None = None
        self.feed(html)
        self.close()

    def source_offset(self) -> int:
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta" and (attributes.get("http-equiv") or "").lower() == (
            "content-security-policy"
        ):
            start = self.source_offset()
            self.policies.append((start, start + len(self.get_starttag_text() or "")))

    def handle_endtag(self, tag: str) -> None:
        if tag == "body":
            self.body_end = self.source_offset()


def reader_view(revision: Revision, policy: str) -> tuple[bytes, str]:
    """Serve only published revisions through the existing public content route."""
    bridge = Path(__file__).with_name("web").joinpath("reader.js").read_bytes()
    digest = base64.b64encode(hashlib.sha256(bridge).digest()).decode("ascii")
    policy = policy.replace("script-src ", f"script-src 'sha256-{digest}' ", 1)
    # 'none' cannot be combined with hashes. There may be no graph renderer.
    policy = policy.replace(f"'sha256-{digest}' 'none'", f"'sha256-{digest}'")
    html = revision.document.html
    markup = _Markup(html)
    if markup.body_end is None:
        raise ContractError("invalid_document")
    script = (
        '<style data-reader-style="1">html{overflow-wrap:anywhere}'
        "img,svg,video,canvas{max-width:100%}"
        ":focus-visible{outline:3px solid Highlight;outline-offset:3px}"
        "@media(prefers-reduced-motion:reduce){*,*::before,*::after{"
        "animation:none!important;transition:none!important;"
        "scroll-behavior:auto!important}}</style>"
        '<script data-reader-document="'
        + escape(revision.document.document_id, quote=True)
        + '" data-reader-revision="'
        + revision.revision_id
        + '">'
        + bridge.decode("utf-8")
        + "</script>"
    )
    changes = [*[(start, end, "") for start, end in markup.policies]]
    changes.append((markup.body_end, markup.body_end, script))
    for start, end, value in sorted(changes, reverse=True):
        html = html[:start] + value + html[end:]
    return html.encode("utf-8"), policy
