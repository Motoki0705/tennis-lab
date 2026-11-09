"""Closed, operator-owned control UI assets for ``--ui explain_html.web:files``."""

from pathlib import Path


def files() -> dict[str, bytes]:
    root = Path(__file__).with_name("web")
    return {
        name: (root / name).read_bytes()
        for name in ("index.html", "app.js", "app.css", "icon.svg")
    }
