"""Deterministic substitutions followed by a full, non-truncating public check."""

from __future__ import annotations

import codecs
import ipaddress
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from board.safety import ABSOLUTE_PATH, SESSION_TOKEN, decoded_text

from .journal import PublicationError, require

TOKEN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{20,})\b"
    r"|\b(?:token|password|secret|api[_-]?key)\s*[:=]",
    re.IGNORECASE,
)
URL = re.compile(r"https?://[^\s<>\"'`\)\]]+", re.IGNORECASE)
ATTACHED_OPTION_PATH = re.compile(
    r"""(?:^|[\s"'`;&|()<>])-[^\s"'`;&|()<>/\\]*?[A-Za-z0-9]"""
    r"(?P<path>[a-zA-Z]:[/\\]|~[/\\]|[/\\])"
)
PATH_END = re.compile(r"[\s<>\"'`()\[\]{};,。、「」『』（）【】]")
PERCENT_BYTES = re.compile(r"(?:%[0-9a-fA-F]{2})+")
PATH_BRACKETS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "（": "）",
    "【": "】",
    "「": "」",
    "『": "』",
}


@dataclass(frozen=True)
class Sanitized:
    text: str
    replacements: int
    local_paths: dict[str, int]


def decoded(value: str) -> str:
    try:
        return decoded_text(value)
    except ValueError:
        raise PublicationError("ambiguous_encoding") from None


def path_view(value: str) -> tuple[str, list[tuple[int, int]]]:
    """Map decoded/NFKC path characters back to the original text's offsets."""
    offsets = [(i, i + 1) for i in range(len(value))]
    for _ in range(4):
        if unquote(value) == value:
            break
        parts: list[str] = []
        mapped: list[tuple[int, int]] = []
        previous = 0
        for match in PERCENT_BYTES.finditer(value):
            parts.append(value[previous : match.start()])
            mapped.extend(offsets[previous : match.start()])
            raw = bytes.fromhex(match[0].replace("%", ""))
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            first = match.start()
            for i, byte in enumerate(raw):
                end = match.start() + 3 * (i + 1)
                chunk = decoder.decode(bytes([byte]), final=i == len(raw) - 1)
                if chunk:
                    parts.append(chunk)
                    mapped.extend(
                        [(offsets[first][0], offsets[end - 1][1])] * len(chunk)
                    )
                    first = end
            previous = match.end()
        parts.append(value[previous:])
        mapped.extend(offsets[previous:])
        value, offsets = "".join(parts), mapped
    require(unquote(value) == value, "ambiguous_encoding")
    parts, mapped = [], []
    for char, span in zip(value, offsets, strict=True):
        normalized = unicodedata.normalize("NFKC", char)
        parts.append(normalized)
        mapped.extend([span] * len(normalized))
    return "".join(parts), mapped


def path_end(value: str, start: int) -> int:
    quote_char = value[start - 1] if start else ""
    if quote_char in {'"', "'", "`"}:
        line_end = re.search(r"[\r\n]", value[start:])
        limit = start + line_end.start() if line_end else len(value)
        closing = value.find(quote_char, start, limit)
        if closing >= 0:
            return closing
    expected: list[str] = []
    end = start
    while end < len(value):
        # Known identities may already have been replaced inside a path.
        if value.startswith("[private identifier]", end):
            end += len("[private identifier]")
            continue
        char = value[end]
        if char in PATH_BRACKETS:
            expected.append(PATH_BRACKETS[char])
        elif expected and char == expected[-1]:
            expected.pop()
        elif PATH_END.match(value, end):
            break
        end += 1
    return end


def mask_local_paths(text: str) -> tuple[str, dict[str, int]]:
    view, offsets = path_view(text)
    urls = [(m.start(), m.end()) for m in URL.finditer(view)]
    starts = {m.start() for m in ABSOLUTE_PATH.finditer(view)}
    starts.update(m.start("path") for m in ATTACHED_OPTION_PATH.finditer(view))
    if URL.sub("", view).strip().strip("\"'`") == "/":
        starts.update(i for i, c in enumerate(view) if c == "/")
    spans: list[tuple[int, int, str]] = []
    for start in sorted(starts):
        if any(left <= start < right for left, right in urls):
            continue
        end = path_end(view, start)
        prefix = view[start : start + 2]
        if prefix.startswith("~"):
            kind = "home"
        elif prefix == "\\\\":
            kind = "unc"
        elif prefix.startswith("\\") or len(prefix) == 2 and prefix[1] == ":":
            kind = "windows"
        else:
            kind = "posix"
        left, right = offsets[start][0], offsets[end - 1][1]
        if spans and left < spans[-1][1]:
            previous = spans[-1]
            spans[-1] = (previous[0], max(previous[1], right), previous[2])
        else:
            spans.append((left, right, kind))
    counts: Counter[str] = Counter(kind for _, _, kind in spans)
    for start, end, _ in reversed(spans):
        text = text[:start] + "[local path]" + text[end:]
    return text, dict(sorted(counts.items()))


class Guard:
    def __init__(
        self,
        private: Iterable[str] = (),
        roots: Iterable[str] = (),
        *,
        repo: str | None = None,
        commit: str | None = None,
    ) -> None:
        self.private = {v for v in private if v}
        self.roots = sorted({p.rstrip("/") for p in roots if p}, key=len, reverse=True)
        self.repo, self.commit = repo, commit

    def check(self, value: Any, *, allow_local_paths: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                self.check(key, allow_local_paths=allow_local_paths)
                self.check(item, allow_local_paths=allow_local_paths)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.check(item, allow_local_paths=allow_local_paths)
        elif isinstance(value, str):
            text = unicodedata.normalize("NFKC", decoded(value))
            require(not any(p in text for p in self.private), "private_value")
            require(not TOKEN.search(text), "credential")
            require(not SESSION_TOKEN.search(text), "private_identifier")
            require(
                not any(
                    unicodedata.category(c).startswith("C") and c not in "\n\r\t"
                    for c in text
                ),
                "control_character",
            )
            # Inspect raw URLs too: decoding an encoded quote/bracket before
            # finding URLs can otherwise move a private query path outside one.
            urls = dict.fromkeys(
                match[0]
                for representation in (value, text)
                for match in URL.finditer(representation)
            )
            for raw_url in urls:
                try:
                    url = urlsplit(raw_url)
                    host = url.hostname
                    require(
                        bool(host) and not url.username and not url.password,
                        "unsafe_url",
                    )
                    require(
                        host not in {"localhost", "localhost.localdomain"}, "unsafe_url"
                    )
                    require(
                        not str(host).endswith((".local", ".internal", ".localhost")),
                        "unsafe_url",
                    )
                    require(
                        not ABSOLUTE_PATH.search(
                            unicodedata.normalize("NFKC", decoded(url.query))
                        )
                        and not ABSOLUTE_PATH.search(
                            unicodedata.normalize("NFKC", decoded(url.fragment))
                        ),
                        "unsafe_url",
                    )
                    try:
                        address = ipaddress.ip_address(str(host))
                    except ValueError:
                        pass
                    else:
                        require(address.is_global, "unsafe_url")
                except ValueError:
                    raise PublicationError("unsafe_url") from None
            paths = URL.sub("", text)
            if not allow_local_paths:
                require(
                    not ABSOLUTE_PATH.search(paths)
                    and paths.strip().strip("\"'`") != "/",
                    "absolute_path",
                )
                require(not ATTACHED_OPTION_PATH.search(paths), "absolute_path")

    def sanitize(self, text: str) -> Sanitized:
        require(isinstance(text, str) and bool(text.strip()), "empty_text")
        count = 0
        # Only known checkout roots have a meaningful, unambiguous public form.
        for root in self.roots:

            def reference(match: re.Match[str]) -> str:
                require(bool(self.repo) and bool(self.commit), "reference_unavailable")
                target = match[1].rstrip(">")
                require(".." not in target.split("/"), "unsafe_reference")
                line = re.fullmatch(r"(.+?):([1-9]\d*)(?::\d+)?", target)
                filename = line[1] if line else target
                anchor = "#L" + line[2] if line else ""
                return f"](https://github.com/{self.repo}/blob/{self.commit}/{quote(filename, safe='/')}{anchor})"

            text, n = re.subn(
                r"\]\(<?" + re.escape(root) + r"/([^\)\r\n]+)\)", reference, text
            )
            count += n
            text, n = re.subn(re.escape(root) + r"(?=/|[\s\"'`\)\]]|$)", ".", text)
            count += n
        for private in sorted(self.private, key=len, reverse=True):
            if private in self.roots:
                continue
            n = text.count(private)
            text = text.replace(private, "[private identifier]")
            count += n
        # Do not let a path replacement swallow credentials or unsafe URLs.
        self.check(text, allow_local_paths=True)
        text, paths = mask_local_paths(text)
        count += sum(paths.values())
        self.check(text)
        return Sanitized(text, count, paths)
