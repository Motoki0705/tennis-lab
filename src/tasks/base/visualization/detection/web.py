"""Local image review APIs with queued, short-lived GPU inference."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.tasks.base.visualization.inference_queue import run_queued_inference

STATIC = Path(__file__).parent / "static"
DetectionTask = Literal["ball_detection", "court_detection"]


class DetectionBackend(Protocol):
    """Image-space service implemented by each detection task."""

    def catalog(self) -> dict[str, Any]: ...

    def scenes(
        self,
        dataset: str,
        search: str = "",
        offset: int = 0,
        limit: int = 100,
        checkpoint: str | None = None,
    ) -> dict[str, Any]: ...

    def preview(self, scene: str, start: int = 0, count: int = 1) -> dict[str, Any]: ...

    def image(self, scene: str, frame: int) -> bytes: ...

    def validate(
        self,
        checkpoint: str,
        scene: str,
        start: int = 0,
        count: int = 1,
        threshold: float = 0.5,
        device: str = "cuda",
    ) -> Any: ...

    def infer(
        self,
        checkpoint: str,
        scene: str,
        start: int = 0,
        count: int = 1,
        threshold: float = 0.5,
        device: str = "cuda",
    ) -> dict[str, Any]: ...


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: str = Field(min_length=1)
    scene: str = Field(min_length=1)
    start: int = Field(default=0, ge=0)
    count: int = Field(default=1, ge=1, le=64)
    threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    device: Literal["cuda", "cpu"] = "cuda"


def create_detection_app(
    service: DetectionBackend,
    *,
    task: DetectionTask,
    mode: Literal["review", "inference"],
    service_config: dict[str, Any],
) -> FastAPI:
    app = FastAPI(title=f"{task} {mode}", docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )
    inference_lock = threading.Lock()

    @app.middleware("http")
    async def local_requests(request: Request, call_next: Any) -> Response:
        # A local server may read trusted pickle checkpoints and use the GPU.
        origin = request.headers.get("origin")
        if request.method == "POST" and origin is not None:
            from urllib.parse import urlsplit

            if urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse(
                    {"detail": "Cross-origin inference is forbidden."}, status_code=403
                )
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    def invalid(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(FileNotFoundError)
    def missing(request: Request, error: FileNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.exception_handler(RuntimeError)
    def failed(request: Request, error: RuntimeError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        if name not in {"app.js", "viewer.mjs", "style.css", "icons.mjs"}:
            raise HTTPException(404)
        return FileResponse(
            STATIC / name,
            media_type="text/css" if name.endswith("css") else "text/javascript",
        )

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        return {
            **service.catalog(),
            "mode": mode,
            "cuda_available": torch.cuda.is_available(),
        }

    @app.get("/api/scenes")
    def scenes(
        dataset: str,
        search: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=200),
        checkpoint: str | None = None,
    ) -> dict[str, Any]:
        return service.scenes(dataset, search, offset, limit, checkpoint)

    @app.get("/api/preview")
    def preview(
        scene: str, start: int = Query(0, ge=0), count: int = Query(1, ge=1, le=64)
    ) -> dict[str, Any]:
        return service.preview(scene, start, count)

    @app.get("/api/image")
    def image(scene: str, frame: int = Query(0, ge=0)) -> Response:
        return Response(service.image(scene, frame), media_type="image/jpeg")

    @app.post("/api/infer")
    def infer(request: InferenceRequest) -> dict[str, Any]:
        if mode != "inference":
            raise HTTPException(403, "Inference is disabled in dataset review mode.")
        if request.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(409, "An inference request is already running.")
        try:
            arguments = request.model_dump()
            service.validate(**arguments)
            if request.device == "cuda":
                payload = run_queued_inference(
                    task, service=service_config, request=arguments
                )
                result: dict[str, Any] = json.loads(payload)
                return result
            return service.infer(**arguments)
        finally:
            inference_lock.release()

    return app
