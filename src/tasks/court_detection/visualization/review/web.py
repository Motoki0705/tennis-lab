"""Task-local routes for Court review assets and catalog-addressed annotations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

STATIC = Path(__file__).parent / "static"


def install_review_routes(
    app: FastAPI, read_annotation: Callable[[str], dict[str, object]]
) -> None:
    def static(name: str) -> FileResponse:
        if name not in {"review.mjs", "review.css"}:
            raise HTTPException(404)
        return FileResponse(
            STATIC / name,
            media_type="text/css" if name.endswith("css") else "text/javascript",
        )

    def annotation(scene: str) -> dict[str, object]:
        return read_annotation(scene)

    app.add_api_route("/task-static/{name}", static, methods=["GET"])
    app.add_api_route("/api/court/annotation", annotation, methods=["GET"])
