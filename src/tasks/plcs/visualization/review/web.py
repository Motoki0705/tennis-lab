"""Local, read-only API and packaged browser assets for ACCAD review."""

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .service import JOINT_COUNT, ReviewService

STATIC = Path(__file__).parent / "static"
ASSETS = ("style.css", "app.js", "scene.mjs")


def create_app(service: ReviewService) -> FastAPI:
    app = FastAPI(title="ACCAD Review", docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @app.exception_handler(ValueError)
    def bad_request(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(RuntimeError)
    def changed(request: Request, error: RuntimeError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(FileNotFoundError)
    def missing(request: Request, error: FileNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": "Motion file is missing."}, status_code=404)

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

    @app.get("/api/motions/{subject}/{motion}")
    def motion_meta(
        subject: str,
        motion: str,
        stride: Annotated[int, Query(ge=1, le=64)] = 1,
    ) -> dict[str, Any]:
        return service.motion_meta(subject, motion, stride=stride)

    @app.get("/api/motions/{subject}/{motion}/joints")
    def motion_joints(
        subject: str,
        motion: str,
        stride: Annotated[int, Query(ge=1, le=64)] = 1,
        revision: Annotated[str | None, Query()] = None,
    ) -> Response:
        loaded = service.motion_joints(
            subject, motion, stride=stride, revision=revision
        )
        payload = loaded.joints.astype("<f4", copy=False).tobytes()
        return Response(
            payload,
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Count": str(loaded.sampled_frame_count),
                "X-Joint-Count": str(JOINT_COUNT),
                "X-Stride": str(loaded.stride),
                "X-Revision": loaded.summary.revision,
            },
        )

    return app
