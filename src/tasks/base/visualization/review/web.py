"""Local, read-only FastAPI app shared by the BLCS and PLCS review UIs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
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

    def buffer(
        self, form: str, scene_id: str, revision: str | None = None
    ) -> bytes: ...


def create_review_app(
    service: SceneReviewService,
    *,
    title: str,
    task: str,
    extra_static: Mapping[str, Path] | None = None,
    extra_stylesheets: Sequence[str] = (),
    extra_modules: Sequence[str] = (),
) -> FastAPI:
    """Bind the common UI, with optional task-owned inspection assets.

    Extra assets cannot replace the common assets. Modules are loaded before
    ``app.js`` so they can subscribe to its scene/frame lifecycle events.
    """
    extensions = dict(extra_static or {})
    for name, path in extensions.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.(?:css|js|mjs)", name) or name in ASSETS:
            raise ValueError(f"Invalid or reserved review asset name: {name!r}.")
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in (*extra_stylesheets, *extra_modules):
        if name not in extensions:
            raise ValueError(f"Review extension asset is not declared: {name!r}.")

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
    def index() -> Response:
        if not extra_stylesheets and not extra_modules:
            return FileResponse(STATIC / "index.html")
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        links = "\n".join(
            f'<link rel="stylesheet" href="/static/{escape(name, quote=True)}" />'
            for name in extra_stylesheets
        )
        modules = "\n".join(
            f'<script type="module" src="/static/{escape(name, quote=True)}"></script>'
            for name in extra_modules
        )
        html = html.replace("</head>", f"{links}\n</head>")
        html = html.replace(
            '<script type="module" src="/static/app.js"></script>',
            f'{modules}\n<script type="module" src="/static/app.js"></script>',
        )
        return HTMLResponse(html)

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        if name in extensions:
            return FileResponse(extensions[name])
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
