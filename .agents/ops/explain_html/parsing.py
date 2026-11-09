"""Shared fail-closed HTML syntax and CSS resource boundaries."""

from __future__ import annotations

import re
from html.parser import HTMLParser

from .schema import ContractError, ErrorCode

_DOCTYPE = re.compile(r"<!doctype[ \t\r\n\f]+html[ \t\r\n\f]*>", re.I | re.ASCII)
_CSS_RESOURCE = re.compile(
    r"(?:url|src|image|(?:-webkit-)?image-set)\s*\(|@import|\\|/\*", re.I | re.ASCII
)


def reject_css_resources(text: str, *, code: ErrorCode = "broken_reference") -> None:
    """URL syntax is outside inline CSS ownership/change accounting."""
    if _CSS_RESOURCE.search(text):
        raise ContractError(code)


class StrictHTMLParser(HTMLParser):
    """Reject ambiguous comments/declarations; never emulate error recovery."""

    CDATA_CONTENT_ELEMENTS = (*HTMLParser.CDATA_CONTENT_ELEMENTS, "textarea", "title")  # type: ignore[misc]

    def parse_comment(self, i: int, report: int = 1) -> int:
        end = self.rawdata.find("-->", i + 4)
        if end < 0:
            raise ContractError("invalid_document")
        content = self.rawdata[i + 4 : end]
        if content.startswith((">", "->")) or "--" in content or content.endswith("-"):
            raise ContractError("invalid_document")
        if report:
            self.handle_comment(content)
        return end + 3

    def parse_html_declaration(self, i: int) -> int:
        if self.rawdata.startswith("<!--", i):
            return self.parse_comment(i)
        match = _DOCTYPE.match(self.rawdata, i)
        if match is None:
            raise ContractError("invalid_document")
        self.handle_decl(self.rawdata[i + 2 : match.end() - 1])
        return match.end()

    def parse_pi(self, i: int) -> int:
        raise ContractError("invalid_document")

    def close(self) -> None:
        # HTMLParser otherwise flushes an incomplete declaration as text at EOF.
        if not self.cdata_elem and re.search(r"<[!?]", self.rawdata):
            raise ContractError("invalid_document")
        super().close()
