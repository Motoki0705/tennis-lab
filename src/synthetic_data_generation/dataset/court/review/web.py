"""Local, read-only API and packaged browser assets."""

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .service import ReviewService

STATIC = Path(__file__).parent / "static"


def create_app(service: ReviewService) -> FastAPI:
    app = FastAPI(title="Court Review", docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @app.exception_handler(ValueError)
    def bad_data(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(RuntimeError)
    def changed(request: Request, error: RuntimeError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(FileNotFoundError)
    def missing(request: Request, error: FileNotFoundError) -> JSONResponse:
        return JSONResponse(
            {"detail": "Scene or dataset file is missing."}, status_code=404
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        if name not in {"style.css", "app.js", "scene.mjs"}:
            raise HTTPException(404)
        return FileResponse(STATIC / name)

    @app.get("/api/scenes")
    def scenes() -> list[dict[str, str]]:
        return service.scenes()

    @app.get("/api/scenes/{scene}")
    def scene_data(scene: str, revision: str) -> Any:
        return service.load(scene, revision)["summary"]

    @app.get("/api/scenes/{scene}/images/{sample}")
    def image(
        scene: str,
        sample: str,
        revision: str,
        width: Annotated[int, Query(ge=0, le=1600)] = 480,
    ) -> Response:
        try:
            content = service.overlay(scene, revision, sample, width)
        except KeyError as error:
            raise HTTPException(404, "Unknown sample.") from error
        return Response(
            content,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
        )

    return app
