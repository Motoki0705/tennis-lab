"""Local browser application for one explicitly selected scene."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.synthetic_data_generation.alignment.manual.models import (
    ApplyRequest,
    EditRequest,
    LayoutEdit,
)
from src.synthetic_data_generation.alignment.manual.service import AlignmentEditor

STATIC = Path(__file__).parent / "static"


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layout: LayoutEdit
    camera_index: Annotated[int, Field(ge=0, strict=True)]


def create_app(editor: AlignmentEditor) -> FastAPI:
    app = FastAPI(title="Court Alignment Studio", docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def protect_mutations(request: Request, call_next: Any) -> Any:
        if request.method == "POST" and not secrets.compare_digest(
            request.headers.get("x-editor-token", ""), editor.token
        ):
            return JSONResponse(
                {"detail": "Editor session token is missing or invalid."},
                status_code=403,
            )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ValueError)
    async def bad_input(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(RuntimeError)
    async def conflict(request: Request, error: RuntimeError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        if name not in {"editor.js", "style.css", "resize.mjs"}:
            raise HTTPException(404)
        return FileResponse(STATIC / name)

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return editor.state()

    @app.get("/api/heatmap")
    def heatmap() -> FileResponse:
        return FileResponse(editor.owner / "line-heatmaps" / "weighted-projection.png")

    @app.get("/api/cameras/{index}/image")
    def camera_image(index: int) -> FileResponse:
        if not 0 <= index < len(editor.cameras):
            raise HTTPException(404)
        path = (editor.camera_export / editor.cameras[index]["image"]).resolve(
            strict=True
        )
        if (
            not path.is_relative_to(editor.camera_export.resolve())
            or not path.is_file()
        ):
            raise HTTPException(400, "Camera image escapes reconstruction export.")
        return FileResponse(path)

    @app.post("/api/preview")
    def preview(request: PreviewRequest) -> dict[str, Any]:
        if request.camera_index >= len(editor.cameras):
            raise HTTPException(400, "Unknown camera index.")
        return editor.preview(request.layout, request.camera_index)

    @app.post("/api/draft")
    def draft(request: EditRequest) -> dict[str, str]:
        return editor.save_draft(request)

    @app.post("/api/apply")
    def apply(request: ApplyRequest) -> dict[str, Any]:
        return editor.apply(request)

    return app
