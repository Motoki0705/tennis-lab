"""Local GET-only endpoints for stored Court Side review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from src.tasks.court_side.review.service import ReviewService

STATIC = Path(__file__).parent / "static"


def create_app(service: ReviewService) -> FastAPI:
    app = FastAPI(title="Court Side dataset review", docs_url=None, redoc_url=None)

    def home() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    def asset(name: str) -> FileResponse:
        if name not in {"app.js", "style.css"}:
            raise HTTPException(404, "Unknown static asset")
        return FileResponse(STATIC / name)

    def catalog() -> dict[str, Any]:
        try:
            return service.catalog()
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    def frame(case: int = Query(ge=0), frame: int = Query(ge=0)) -> dict[str, Any]:
        try:
            return service.frame(case, frame)
        except IndexError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    def timeline(case: int = Query(ge=0)) -> dict[str, Any]:
        try:
            return service.timeline(case)
        except IndexError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    def image(case: int, camera: str, frame: int) -> Response:
        try:
            return Response(service.image(case, camera, frame), media_type="image/jpeg")
        except (IndexError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except OSError as error:
            raise HTTPException(422, str(error)) from error

    app.add_api_route("/", home, methods=["GET"])
    app.add_api_route("/static/{name}", asset, methods=["GET"])
    app.add_api_route("/api/catalog", catalog, methods=["GET"])
    app.add_api_route("/api/frame", frame, methods=["GET"])
    app.add_api_route("/api/timeline", timeline, methods=["GET"])
    app.add_api_route("/api/image/{case}/{camera}/{frame}", image, methods=["GET"])
    return app
