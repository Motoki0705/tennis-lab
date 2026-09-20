"""Local, read-only FastAPI app shared by the BLCS and PLCS review UIs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.tasks.base.visualization.web_assets import mount_scene_assets

STATIC = Path(__file__).parent / "static"
ASSETS = ("style.css", "app.js", "scene.mjs", "model.mjs")


class SceneReviewService(Protocol):
    """The service surface every task-specific review service provides."""

    def catalog(self) -> dict[str, Any]: ...

    def scenes(self, form: str) -> dict[str, Any]: ...

    def scene(
        self, form: str, scene_id: str, revision: str | None = None
    ) -> dict[str, Any]: ...

    def buffer(self, form: str, scene_id: str, revision: str | None = None) -> bytes: ...


def create_review_app(
    service: SceneReviewService, *, title: str, task: str
) -> FastAPI:
    app = FastAPI(title=title, docs_url=None, redoc_url=None)
    mount_scene_assets(app)
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
        return JSONResponse({"detail": "Scene file is missing."}, status_code=404)

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
    def scenes(form: Annotated[str, Query()]) -> dict[str, Any]:
        return service.scenes(form)

    @app.get("/api/scene")
    def scene(
        form: Annotated[str, Query()],
        scene: Annotated[str, Query()],
        revision: Annotated[str | None, Query()] = None,
    ) -> dict[str, Any]:
        return service.scene(form, scene, revision)

    @app.get("/api/scene/buffer")
    def scene_buffer(
        form: Annotated[str, Query()],
        scene: Annotated[str, Query()],
        revision: Annotated[str | None, Query()] = None,
    ) -> Response:
        payload = service.buffer(form, scene, revision)
        return Response(
            payload,
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
                "X-Task": task,
            },
        )

    return app


__all__ = ["ASSETS", "STATIC", "SceneReviewService", "create_review_app"]
