"""Task-local HTTP UI with read-only, manifest-bound dataset access."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .data import DatasetReview

STATIC = Path(__file__).parent / "static"


def create_app(service: DatasetReview) -> FastAPI:
    app = FastAPI(title="Ball Refiner 3D Dataset Review")

    def invalid_data(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    def javascript() -> FileResponse:
        return FileResponse(STATIC / "app.js", media_type="text/javascript")

    def style() -> FileResponse:
        return FileResponse(STATIC / "style.css", media_type="text/css")

    def catalog() -> JSONResponse:
        return JSONResponse(service.catalog())

    def rallies(dataset: str, split: str) -> JSONResponse:
        return JSONResponse(service.rallies(dataset, split))

    def sequence(dataset: str, rally: str) -> JSONResponse:
        return JSONResponse(service.sequence(dataset, rally))

    def frame(dataset: str, rally: str, frame: int) -> JSONResponse:
        return JSONResponse(service.frame(dataset, rally, frame))

    app.add_exception_handler(ValueError, invalid_data)
    app.add_exception_handler(OSError, invalid_data)
    for path, endpoint in (
        ("/", index),
        ("/app.js", javascript),
        ("/style.css", style),
        ("/api/catalog", catalog),
        ("/api/rallies", rallies),
        ("/api/sequence", sequence),
        ("/api/frame", frame),
    ):
        app.add_api_route(path, endpoint, methods=["GET"])
    return app
