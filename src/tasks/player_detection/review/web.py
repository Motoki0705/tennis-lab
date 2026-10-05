"""Loopback HTTP viewer with GET-only access to existing player annotations."""

# FastAPI's decorators are skipped by the repository's follow-imports policy.
# mypy: disable-error-code="untyped-decorator"

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.tasks.player_detection.review.service import PlayerReviewService

STATIC = Path(__file__).parent / "static"


def create_app(service: PlayerReviewService) -> FastAPI:
    app = FastAPI(title="Player Dataset Review", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def readonly(request: Request, call_next: Any) -> Response:
        if request.method not in {"GET", "HEAD"}:
            return JSONResponse({"detail": "This viewer is read-only."}, status_code=405)
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(FileNotFoundError)
    def missing(request: Request, error: FileNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.exception_handler(ValueError)
    def invalid(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        if name not in {"app.js", "style.css"}:
            raise HTTPException(404)
        return FileResponse(STATIC / name)

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        return service.catalog()

    @app.get("/api/clips")
    def clips(dataset: str, search: str = "", split: str = "all", source: str = "all", flag: str = "all") -> dict[str, Any]:
        return service.clips(dataset, search=search, split=split, source=source, flag=flag)

    @app.get("/api/clip")
    def clip(dataset: str, clip: str) -> dict[str, Any]:
        return service.review(dataset).timeline(clip)

    @app.get("/api/frame")
    def frame(dataset: str, clip: str, frame: int = Query(ge=0)) -> dict[str, Any]:
        return service.review(dataset).frame(clip, frame)

    @app.get("/api/image")
    def image(dataset: str, clip: str, frame: int = Query(ge=0)) -> Response:
        return Response(service.review(dataset).image(clip, frame), media_type="image/jpeg")

    return app
