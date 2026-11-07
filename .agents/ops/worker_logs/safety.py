"""Deterministic substitutions followed by a full, non-truncating public check."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from board.safety import ABSOLUTE_PATH, SESSION_TOKEN

from .journal import PublicationError, require

TOKEN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{20,})\b"
    r"|\b(?:token|password|secret|api[_-]?key)\s*[:=]",
    re.IGNORECASE,
)
URL = re.compile(r"https?://[^\s<>\"'`\)\]]+", re.IGNORECASE)
ATTACHED_OPTION_PATH = re.compile(
    r"""(?:^|[\s"'`;&|()<>])-[^\s"'`;&|()<>]*[A-Za-z0-9]/"""
)


def decoded(value: str) -> str:
    for _ in range(4):
        new = unquote(value)
        if new == value:
            return value
        value = new
    require(unquote(value) == value, "ambiguous_encoding")
    return value


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

    def check(self, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                self.check(key)
                self.check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.check(item)
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
            for match in URL.finditer(text):
                try:
                    url = urlsplit(match[0])
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
                        not ABSOLUTE_PATH.search(url.query)
                        and not ABSOLUTE_PATH.search(url.fragment),
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
            require(
                not ABSOLUTE_PATH.search(paths) and paths.strip().strip("\"'`") != "/",
                "absolute_path",
            )
            require(not ATTACHED_OPTION_PATH.search(paths), "absolute_path")

    def sanitize(self, text: str) -> tuple[str, int]:
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
        self.check(text)
        return text, count
