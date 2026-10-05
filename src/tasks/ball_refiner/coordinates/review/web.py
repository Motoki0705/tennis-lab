"""Local review HTTP boundary and packaged browser assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.tasks.ball_refiner.coordinates.review.contracts import ReviewRequest
from src.tasks.ball_refiner.coordinates.review.service import ReviewService
from src.tasks.base.visualization.inference_queue import run_queued_inference
from src.tasks.base.visualization.web_assets import mount_scene_assets

STATIC = Path(__file__).parent / "static"
ASSETS = frozenset({"index.html", "style.css", "app.mjs", "plots.mjs", "scene.mjs"})


def create_app(service: ReviewService) -> FastAPI:
    app = FastAPI(title="Ball Refiner Review", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    mount_scene_assets(app)

    def invalid(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=422)

    def changed(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    def absent(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=404)

    app.add_exception_handler(ValueError, invalid)
    app.add_exception_handler(RuntimeError, changed)
    app.add_exception_handler(FileNotFoundError, absent)

    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    def asset(name: str) -> FileResponse:
        if name not in ASSETS:
            raise HTTPException(404)
        media = "text/javascript" if name.endswith(".mjs") else "text/css" if name.endswith(".css") else "text/html"
        return FileResponse(STATIC / name, media_type=media, headers={"Cache-Control": "no-cache"})

    def catalog(refresh: bool = False) -> dict[str, Any]:
        return service.catalog(refresh=refresh)

    def preview(body: ReviewRequest) -> dict[str, Any]:
        return service.preview(body)

    def saved(body: ReviewRequest) -> dict[str, Any]:
        return service.saved(body)

    def infer(body: ReviewRequest) -> dict[str, Any]:
        service.validate(body)
        if body.device == "cpu":
            return service.infer(body)
        payload = run_queued_inference("ball_refiner_coordinates", service={
            "data_root": str(service.data_root), "outputs_root": str(service.outputs_root),
            "checkpoints_root": str(service.checkpoints_root)}, request=body.model_dump())
        result: dict[str, Any] = json.loads(payload)
        return result

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/static/{name}", asset, methods=["GET"])
    app.add_api_route("/api/catalog", catalog, methods=["GET"])
    app.add_api_route("/api/preview", preview, methods=["POST"])
    app.add_api_route("/api/saved", saved, methods=["POST"])
    app.add_api_route("/api/infer", infer, methods=["POST"])
    return app
