"""Loopback-only, read-only board web entry point. Imported only by board web."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .errors import BoardError
from .web_data import BoardReader, InvalidQuery, MissingPost

STATIC = Path(__file__).parent / "static"
ASSETS = {"app.js", "style.css"}
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
)


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        # Preserve JSON text values while also making the wire representation
        # safe from interpretation as HTML. The UI uses textContent throughout.
        payload: bytes = super().render(content)
        return (
            payload.replace(b"<", b"\\u003c")
            .replace(b">", b"\\u003e")
            .replace(b"&", b"\\u0026")
        )


def create_app(env: Mapping[str, str]) -> FastAPI:
    reader = BoardReader(env)
    app = FastAPI(
        title="親間掲示板",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        default_response_class=SafeJSONResponse,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost"],
        www_redirect=False,
    )

    async def boundary(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method != "GET":
            response: Response = JSONResponse(
                {"detail": "Read-only board: GET required."},
                status_code=405,
                headers={"Allow": "GET"},
            )
        else:
            response = await call_next(request)
        response.headers.update(
            {
                "Content-Security-Policy": CSP,
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
            }
        )
        return response

    app.middleware("http")(boundary)

    def unavailable(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            {
                "detail": "Board data unavailable or inconsistent; check the local configuration."
            },
            status_code=503,
        )

    def invalid(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": "Invalid board query."}, status_code=422)

    def missing(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": "Board post not found."}, status_code=404)

    for exception in (BoardError, OSError, KeyError, TypeError, ValueError):
        app.add_exception_handler(exception, unavailable)
    app.add_exception_handler(InvalidQuery, invalid)
    app.add_exception_handler(RequestValidationError, invalid)
    app.add_exception_handler(MissingPost, missing)

    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    def static(name: str) -> Response:
        if name not in ASSETS:
            return JSONResponse({"detail": "Asset not found."}, status_code=404)
        return FileResponse(STATIC / name)

    def changes(
        cursor: str | None = None, name: str = "", direction: str = "both"
    ) -> dict[str, Any]:
        return reader.changes(cursor=cursor, name=name, direction=direction)

    def thread(post_id: str) -> dict[str, Any]:
        return reader.thread(post_id)

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/static/{name}", static, methods=["GET"])
    app.add_api_route("/api/changes", changes, methods=["GET"])
    app.add_api_route("/api/thread/{post_id}", thread, methods=["GET"])
    return app


def serve(env: Mapping[str, str], *, port: int = 8790) -> None:
    import uvicorn

    # No host override: WSL localhost forwarding works with the loopback bind.
    # Disable access logs so arbitrary query strings are not logged as public text.
    uvicorn.run(create_app(env), host="127.0.0.1", port=port, access_log=False)
