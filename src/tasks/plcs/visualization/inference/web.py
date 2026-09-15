"""Local HTTP API and packaged browser assets for the PLCS inference UI."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.tasks.base.visualization.inference_queue import run_queued_inference
from src.tasks.base.visualization.web_assets import mount_scene_assets

from .service import (
    DEFAULT_SPLIT,
    MAX_SCENE_LIMIT,
    SPLITS,
    InferenceService,
    PredictionRequest,
    PredictionResult,
)

STATIC = Path(__file__).parent / "static"
ASSETS = ("style.css", "app.js", "court_scene.mjs")


class PredictBody(BaseModel):
    """Validated body for one inference run."""

    checkpoint: str = Field(min_length=1)
    family: str = Field(min_length=1)
    scene: str = Field(min_length=1)
    cameras: list[int] = Field(min_length=1)
    reference_camera_id: str | None = None
    window_start: int = Field(default=0, ge=0)
    window_length: int = Field(default=1, ge=1)
    canonical_pose_source: str = "gt"
    device: str | None = None


def _resolve_body(service: InferenceService, body: PredictBody) -> PredictionRequest:
    return PredictionRequest(
        checkpoint=body.checkpoint,
        family=body.family,
        scene=body.scene,
        cameras=tuple(body.cameras),
        reference_camera_id=body.reference_camera_id,
        window_start=body.window_start,
        window_length=body.window_length,
        canonical_pose_source=body.canonical_pose_source,
        device=body.device or service.device,
    )


def _binary_response(result: PredictionResult) -> Response:
    payload = result.to_bytes()
    return Response(
        payload,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Payload-Bytes": str(len(payload)),
        },
    )


def _validation_detail(error: RequestValidationError) -> str:
    problems = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ()))
        problems.append(f"{location}: {item.get('msg', 'invalid value')}")
    return "; ".join(problems) or "invalid request body."


def create_app(service: InferenceService) -> FastAPI:
    """Build the FastAPI application for one configured service."""
    app = FastAPI(title="PLCS Inference", docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )
    # The 3D viewport uses the shared packaged Three.js modules.
    mount_scene_assets(app)

    @app.exception_handler(RequestValidationError)
    def invalid_body(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": _validation_detail(error)}, status_code=422)

    @app.exception_handler(ValueError)
    def bad_request(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(FileNotFoundError)
    def missing(request: Request, error: FileNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.exception_handler(RuntimeError)
    def changed(request: Request, error: RuntimeError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        if name not in ASSETS:
            raise HTTPException(404)
        return FileResponse(STATIC / name)

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        return service.catalog()

    @app.get("/api/scenes")
    def scenes(
        family: str,
        split: Annotated[str, Query()] = DEFAULT_SPLIT,
        query: Annotated[str, Query()] = "",
        limit: Annotated[int, Query(ge=1, le=MAX_SCENE_LIMIT)] = 200,
    ) -> dict[str, Any]:
        if split not in SPLITS:
            raise HTTPException(422, f"split must be one of {list(SPLITS)}.")
        return service.scenes(family, split=split, query=query, limit=limit)

    @app.get("/api/scenes/{family}/{scene}")
    def scene_detail(
        family: str,
        scene: str,
        checkpoint: Annotated[str | None, Query()] = None,
    ) -> dict[str, Any]:
        return service.scene_detail(family, scene, checkpoint=checkpoint)

    @app.get("/api/scenes/{family}/{scene}/preview")
    def scene_preview(
        family: str,
        scene: str,
        cameras: Annotated[list[int] | None, Query()] = None,
        window_start: Annotated[int, Query(ge=0)] = 0,
        window_length: Annotated[int | None, Query(ge=1)] = None,
        reference_camera_id: Annotated[str | None, Query()] = None,
    ) -> Response:
        result = service.scene_preview(
            family,
            scene,
            cameras=tuple(cameras) if cameras else None,
            window_start=window_start,
            window_length=window_length,
            reference_camera_id=reference_camera_id,
        )
        return _binary_response(result)

    @app.post("/api/validate")
    def validate(body: PredictBody) -> dict[str, Any]:
        """Validate a request without touching the GPU; used before enqueueing."""
        service.validate_prediction_request(_resolve_body(service, body))
        return {"valid": True}

    @app.post("/api/predict")
    def predict(body: PredictBody) -> Response:
        request = _resolve_body(service, body)
        validated = service.validate_prediction_request(request)
        if str(validated["device"]).split(":")[0] == "cpu":
            return _binary_response(service.predict(request))
        arguments = asdict(request)
        arguments["device"] = validated["device"]
        payload = run_queued_inference(
            "plcs",
            service={
                "data_root": str(service.data_root),
                "checkpoint_root": str(service.checkpoint_root),
                "checkpoint_roots": [
                    str(root) for root in service.extra_checkpoint_roots
                ],
                "project_root": str(service._resolver.roots.project_root),
                "camera_depth": service.camera_depth,
                "device": validated["device"],
            },
            request=arguments,
        )
        return Response(
            payload,
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-Payload-Bytes": str(len(payload)),
            },
        )

    return app


__all__ = ["ASSETS", "PredictBody", "STATIC", "create_app", "mount_scene_assets"]
