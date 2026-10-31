"""Loopback board web entry point with explicit, CSRF-protected human writes."""

from __future__ import annotations

import re
import secrets
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import messages, notifiers, reactions, runtime
from .delivery import Notifiers
from .errors import BoardError
from .execution_data import ExecutionReader
from .web_data import BoardReader, InvalidQuery, MissingPost

STATIC = Path(__file__).parent / "static"
ASSETS = {"app.js", "execution.js", "style.css"}
WRITE_PATH = re.compile(r"/api/posts(?:/[^/]+/(?:replies|read|reactions/[^/]+))?")
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


class PostInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=60000)
    kind: str
    to: list[str] = Field(default_factory=list)


class ReplyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    body: str = Field(min_length=1, max_length=60000)
    to: list[str] = Field(default_factory=list)


def create_app(
    env: Mapping[str, str], *, notification_routes: Notifiers | None = None
) -> FastAPI:
    env = dict(env)
    reader = BoardReader(env)
    execution_reader = ExecutionReader(reader)
    routes = (
        notifiers.configured(env)
        if notification_routes is None
        else notification_routes
    )
    csrf_token = secrets.token_urlsafe(32)
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
        match = WRITE_PATH.fullmatch(request.url.path)
        allowed = {"GET"}
        if match:
            allowed = (
                {"PUT", "DELETE"} if "/reactions/" in request.url.path else {"POST"}
            )
        if request.method not in allowed:
            response: Response = JSONResponse(
                {"detail": "Board method not allowed."},
                status_code=405,
                headers={"Allow": ", ".join(sorted(allowed))},
            )
        elif request.method != "GET" and (
            request.headers.getlist("origin")
            != [f"{request.url.scheme}://{request.url.netloc}"]
            or len(request.headers.getlist("x-csrf-token")) != 1
            or not secrets.compare_digest(
                request.headers.get("x-csrf-token", "").encode(), csrf_token.encode()
            )
        ):
            response = JSONResponse(
                {"detail": "Invalid board request origin or CSRF token."},
                status_code=403,
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

    def index() -> HTMLResponse:
        page = (STATIC / "index.html").read_text(encoding="utf-8")
        page = page.replace(
            "</head>", f'<meta name="board-csrf-token" content="{csrf_token}">\n</head>'
        )
        return HTMLResponse(page, headers={"X-CSRF-Token": csrf_token})

    def static(name: str) -> Response:
        if name not in ASSETS:
            return JSONResponse({"detail": "Asset not found."}, status_code=404)
        return FileResponse(STATIC / name)

    def changes(
        cursor: str | None = None,
        name: str = "",
        direction: str = "both",
        refresh_roster: bool = False,
    ) -> dict[str, Any]:
        return reader.changes(
            cursor=cursor, name=name, direction=direction, refresh_roster=refresh_roster
        )

    def thread(post_id: str) -> dict[str, Any]:
        return reader.thread(post_id)

    def execution(response: Response, refresh: bool = False) -> dict[str, Any]:
        result = execution_reader.read(refresh=refresh)
        if refresh and result["github_refresh"]["status"] == "running":
            response.status_code = 202
        return result

    def post(payload: PostInput) -> dict[str, Any]:
        return messages.post(env, **payload.model_dump(), human=True, notifiers=routes)

    def reply(post_id: str, payload: ReplyInput) -> dict[str, Any]:
        return messages.reply(
            env, post_id=post_id, **payload.model_dump(), human=True, notifiers=routes
        )

    def react(post_id: str, code: str, request: Request) -> dict[str, Any]:
        return reactions.react(
            env,
            post_id=post_id,
            emoji=code,
            human=True,
            remove=request.method == "DELETE",
        )

    def read(post_id: str) -> dict[str, Any]:
        return messages.read(env, post_id, human=True)

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/static/{name}", static, methods=["GET"])
    app.add_api_route("/api/changes", changes, methods=["GET"])
    app.add_api_route("/api/execution", execution, methods=["GET"])
    app.add_api_route("/api/runtime", lambda: runtime.snapshot(env), methods=["GET"])
    app.add_api_route("/api/thread/{post_id}", thread, methods=["GET"])
    app.add_api_route("/api/posts", post, methods=["POST"])
    app.add_api_route("/api/posts/{post_id}/replies", reply, methods=["POST"])
    app.add_api_route(
        "/api/posts/{post_id}/reactions/{code}", react, methods=["PUT", "DELETE"]
    )
    app.add_api_route("/api/posts/{post_id}/read", read, methods=["POST"])
    return app


def serve(env: Mapping[str, str], *, port: int = 8790) -> None:
    import uvicorn

    # No host override: WSL localhost forwarding works with the loopback bind.
    # Disable access logs so arbitrary query strings are not logged as public text.
    uvicorn.run(create_app(env), host="127.0.0.1", port=port, access_log=False)
