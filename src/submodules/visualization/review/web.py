"""Local HTTP endpoints for read-only saved-pose review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.submodules.visualization.review.store import PoseReviewStore


def create_app(store: PoseReviewStore) -> FastAPI:
    app = FastAPI(title="Saved GVHMR / Pose review")
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    def metadata() -> dict[str, Any]:
        store.assert_unchanged()
        return store.metadata()

    def frame(person: int, camera: str, frame: int) -> dict[str, Any]:
        try:
            return store.frame(person, camera, frame)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def timeline(person: int, camera: str) -> dict[str, Any]:
        try:
            store.assert_unchanged()
            if camera not in store.videos:
                raise ValueError("Unknown camera")
            return store.timeline(person, camera)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def image(camera: str, frame: int) -> Response:
        try:
            data = store.image(camera, frame)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/api/meta", metadata, methods=["GET"])
    app.add_api_route("/api/frame", frame, methods=["GET"])
    app.add_api_route("/api/timeline", timeline, methods=["GET"])
    app.add_api_route("/api/image", image, methods=["GET"])
    return app
