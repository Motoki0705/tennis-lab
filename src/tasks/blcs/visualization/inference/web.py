"""Local, read-only inference API and packaged browser assets for BLCS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.tasks.base.visualization.inference_queue import run_queued_inference
from src.tasks.base.visualization.web_assets import mount_scene_assets

from .service import InferenceService

STATIC = Path(__file__).parent / "static"
ASSETS = ("index.html", "style.css", "app.js", "scene.mjs")


class InferRequest(BaseModel):
    """One inference request body encoded by the browser client."""

    checkpoint: str = Field(min_length=1)
    form: str = Field(min_length=1)
    scene: str = Field(min_length=1)
    cameras: list[int] | None = None
    reference_camera_id: str | None = None
    device: str | None = None
    window: int | None = None


def create_app(service: InferenceService) -> FastAPI:
    """Build the FastAPI application for one service instance.

    Routes are registered through ``add_api_route`` rather than decorators so
    the module type-checks under the repository's ``--follow-imports=skip``
    mypy gate without a per-module override.
    """
    app = FastAPI(title="BLCS Inference", docs_url=None, redoc_url=None)
    mount_scene_assets(app)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    def bad_request(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=422)

    def changed(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    def missing(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=404)

    app.add_exception_handler(ValueError, bad_request)
    app.add_exception_handler(RuntimeError, changed)
    app.add_exception_handler(FileNotFoundError, missing)

    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    def static(name: str) -> FileResponse:
        if name not in ASSETS:
            raise HTTPException(404)
        return FileResponse(STATIC / name)

    def catalog() -> dict[str, Any]:
        return service.catalog()

    def scenes(
        form: Annotated[str, Query(min_length=1)],
        split: Annotated[str | None, Query()] = None,
        query: Annotated[str, Query()] = "",
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1)] = 200,
    ) -> dict[str, Any]:
        return service.scenes(
            form=form,
            split=split,
            query=query,
            offset=offset,
            limit=limit,
        )

    def scene(
        form: Annotated[str, Query(min_length=1)],
        scene: Annotated[str, Query(min_length=1)],
    ) -> dict[str, Any]:
        return service.scene(form=form, scene_id=scene)

    def infer(body: InferRequest) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "checkpoint": body.checkpoint,
            "form": body.form,
            "scene_id": body.scene,
            "cameras": body.cameras,
            "reference_camera_id": body.reference_camera_id,
            "device": body.device or service.device,
            "window": body.window,
        }
        validated = service.validate_inference_request(**arguments)
        arguments["device"] = validated.device_key
        device_type = validated.device_key.split(":")[0]
        if device_type not in {"cpu", "cuda"}:
            raise ValueError("The inference UI supports CPU and CUDA devices.")
        if device_type == "cpu":
            return service.infer(**arguments)
        payload = run_queued_inference(
            "blcs",
            service={
                "outputs_root": str(service.outputs_root),
                "checkpoints_root": str(service.checkpoints_root),
                "data_root": str(service.data_root),
                "device": validated.device_key,
            },
            request=arguments,
        )
        result: dict[str, Any] = json.loads(payload)
        return result

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/static/{name}", static, methods=["GET"])
    app.add_api_route("/api/catalog", catalog, methods=["GET"])
    app.add_api_route("/api/scenes", scenes, methods=["GET"])
    app.add_api_route("/api/scene", scene, methods=["GET"])
    app.add_api_route("/api/infer", infer, methods=["POST"])

    return app


__all__ = ["ASSETS", "InferRequest", "create_app"]
