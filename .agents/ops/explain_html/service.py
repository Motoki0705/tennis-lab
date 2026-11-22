"""Loopback HTTP boundary; routes, wire types and statuses come from schema.py."""

from __future__ import annotations

import base64
import logging
import mimetypes
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .jobs import JobController
from .paths import relative_path
from .reader import reader_view
from .schema import (
    CancelRequest,
    ContractError,
    Document,
    DocumentResponse,
    EditRequest,
    ErrorResponse,
    HistoryResponse,
    Model,
    RestoreRequest,
    SnapshotRequest,
    SnapshotResponse,
    contract_schema,
    parse,
)

LOG = logging.getLogger(__name__)
CONTROL_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "frame-src 'self'; img-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
DEFAULT_UI = b"<!doctype html><html><head><meta charset='utf-8'><title>Explain HTML</title></head><body><p>Explain HTML service is running. Configure the browser editor to edit a document.</p></body></html>"


def content_policy(document: Document | None = None) -> str:
    # A trusted bundle validator must authorize every renderer asset before
    # publication. Only those revision-pinned inline bytes can run here.
    hashes = (
        [
            "'sha256-"
            + base64.b64encode(bytes.fromhex(asset.sha256)).decode("ascii")
            + "'"
            for asset in document.assets
            if asset.kind == "renderer"
        ]
        if document
        else []
    )
    script = " ".join(hashes) or "'none'"
    return (
        f"sandbox allow-scripts; default-src 'none'; script-src {script}; "
        "img-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'none'; "
        "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
    )


def public_error(error: ContractError) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(error=error.error).model_dump(mode="json"),
        status_code=error.status_code,
    )


def create_app(
    controller: JobController,
    *,
    port: int,
    token: str | None = None,
    ui_files: Mapping[str, bytes] | None = None,
) -> FastAPI:
    """The fixed trusted UI may receive the token via a launch-URL fragment.

    Generated content never gets a token or cookies. Do not enable CORS, put the
    token in a query string, or grant allow-same-origin to the content sandbox.
    """
    token = token if token is not None else secrets.token_urlsafe(32)
    if len(token) < 32 or not 0 < port <= 65535:
        raise ContractError("invalid_request")
    hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
    if port == 80:
        hosts.update({"localhost", "127.0.0.1"})
    origins = {f"http://{host}" for host in hosts}
    ui = dict(ui_files) if ui_files is not None else {"index.html": DEFAULT_UI}
    for name in ui:
        relative_path(name)
    if "index.html" not in ui:
        raise ContractError("invalid_request")
    contract = contract_schema()
    token_header = contract["components"]["securitySchemes"]["SessionToken"]["name"]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await run_in_threadpool(controller.close)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.session_token = token

    def guard(request: Request, *, authenticated: bool) -> None:
        origin = request.headers.getlist("origin")
        if len(origin) > 1 or (
            origin
            and origin[0] not in origins
            and not (not authenticated and origin[0] == "null")
        ):
            raise ContractError("invalid_origin")
        if authenticated:
            if request.headers.get("sec-fetch-site") == "cross-site":
                raise ContractError("invalid_origin")
            values = request.headers.getlist(token_header)
            if len(values) != 1 or not secrets.compare_digest(
                values[0].encode("utf-8"), token.encode("utf-8")
            ):
                raise ContractError("unauthorized")

    async def boundary(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            host = request.headers.getlist("host")
            if len(host) != 1 or host[0].lower() not in hosts:
                raise ContractError("invalid_host")
            raw = request.scope.get("raw_path", b"")
            path = request.url.path
            if (
                b"%" in raw
                or "\\" in path
                or "//" in path
                or any(part in {".", ".."} for part in path.split("/"))
                or request.url.query
            ):
                raise ContractError("unsafe_path")
            response = await call_next(request)
        except ContractError as exc:
            response = public_error(exc)
        except Exception:
            LOG.exception("Private HTTP failure")
            response = public_error(ContractError("storage_error"))
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            }
        )
        response.headers.setdefault(
            "Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"
        )
        return response

    async def contract_error(request: Request, error: Exception) -> Response:
        assert isinstance(error, ContractError)
        return public_error(error)

    async def http_error(request: Request, error: Exception) -> Response:
        return public_error(ContractError("invalid_request"))

    app.middleware("http")(boundary)
    app.add_exception_handler(ContractError, contract_error)
    app.add_exception_handler(HTTPException, http_error)

    async def fixed_ui(request: Request) -> Response:
        guard(request, authenticated=False)
        name = request.path_params.get("name", "index.html")
        if name not in ui:
            raise ContractError("invalid_request")
        return Response(
            ui[name],
            media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
            headers={"Content-Security-Policy": CONTROL_CSP},
        )

    app.add_api_route("/", fixed_ui, methods=["GET"])
    app.add_api_route("/ui/{name:path}", fixed_ui, methods=["GET"])

    def json_response(result: Model, status: int) -> Response:
        return JSONResponse(result.model_dump(mode="json"), status_code=status)

    async def dispatch(request: Request, operation: dict[str, Any]) -> Response:
        guard(request, authenticated=bool(operation.get("security")))
        parameters: dict[str, str] = request.path_params
        document_id = parameters.get("document_id", "")
        revision_id = parameters.get("revision_id", "")
        request_id = parameters.get("request_id", "")
        response_codes = [key for key in operation["responses"] if key != "default"]
        status = int(response_codes[0])
        if request.method == "POST":
            reference = operation["requestBody"]["content"]["application/json"][
                "schema"
            ]["$ref"].split("/")[-1]
            body = await request.body()
            models: dict[
                str,
                type[EditRequest | CancelRequest | RestoreRequest | SnapshotRequest],
            ] = {
                "EditRequest": EditRequest,
                "CancelRequest": CancelRequest,
                "RestoreRequest": RestoreRequest,
                "SnapshotRequest": SnapshotRequest,
            }
            payload = parse(models[reference], body)
            if payload.document_id != document_id or (
                request_id and getattr(payload, "request_id", None) != request_id
            ):
                raise ContractError("invalid_request")
            if isinstance(payload, EditRequest):
                result: Model = await run_in_threadpool(controller.submit, payload)
            elif isinstance(payload, CancelRequest):
                result = await run_in_threadpool(
                    controller.cancel, document_id, request_id
                )
            elif isinstance(payload, RestoreRequest):
                result = DocumentResponse(
                    revision=await run_in_threadpool(controller.restore, payload)
                )
            else:
                assert isinstance(payload, SnapshotRequest)
                result = await run_in_threadpool(
                    controller.store.snapshot, document_id, payload.revision_id
                )
            return json_response(result, status)

        if "snapshot_id" in parameters:
            snapshot_id, name = parameters["snapshot_id"], parameters["file_path"]
            data = await run_in_threadpool(
                controller.store.snapshot_file, snapshot_id, name
            )
            # The snapshot manifest pins the document and revision for CSP too.
            snapshot = parse(
                SnapshotResponse,
                await run_in_threadpool(
                    controller.store.snapshot_file, snapshot_id, "manifest.json"
                ),
            )
            revision = await run_in_threadpool(
                controller.store.revision, snapshot.document_id, snapshot.revision_id
            )
            return Response(
                data,
                media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                headers={"Content-Security-Policy": content_policy(revision.document)},
            )
        if "file_name" in parameters or "asset_name" in parameters:
            revision = await run_in_threadpool(
                controller.store.revision, document_id, revision_id
            )
            if "asset_name" in parameters:
                asset = next(
                    (
                        item
                        for item in revision.document.assets
                        if item.path == f"assets/{parameters['asset_name']}"
                    ),
                    None,
                )
                if asset is None:
                    raise ContractError("broken_reference")
                data = await run_in_threadpool(
                    controller.store.asset_bytes,
                    document_id,
                    revision_id,
                    asset.asset_id,
                )
                media_type = asset.media_type
            else:
                name = parameters["file_name"]
                if name == "reader.html":
                    data, policy = reader_view(
                        revision, content_policy(revision.document)
                    )
                    return Response(
                        data,
                        media_type="text/html",
                        headers={"Content-Security-Policy": policy},
                    )
                data = await run_in_threadpool(
                    controller.store.content_file, document_id, revision_id, name
                )
                media_type = "text/html" if name == "index.html" else "application/json"
            return Response(
                data,
                media_type=media_type,
                headers={"Content-Security-Policy": content_policy(revision.document)},
            )
        if request_id:
            result = await run_in_threadpool(controller.status, document_id, request_id)
        elif revision_id:
            result = DocumentResponse(
                revision=await run_in_threadpool(
                    controller.store.revision, document_id, revision_id
                )
            )
        elif request.url.path.endswith("/revisions"):
            result = HistoryResponse(
                document_id=document_id,
                revisions=await run_in_threadpool(
                    controller.store.history, document_id
                ),
            )
        else:
            result = DocumentResponse(
                revision=await run_in_threadpool(controller.store.current, document_id)
            )
        return json_response(result, status)

    # Register the actual wire paths/methods from the common contract, rather
    # than maintaining a second route/status/security table in the service.
    for path, operations in contract["paths"].items():
        for method, operation in operations.items():

            def endpoint_for(
                specification: dict[str, Any],
            ) -> Callable[[Request], Awaitable[Response]]:
                async def endpoint(request: Request) -> Response:
                    return await dispatch(request, specification)

                return endpoint

            app.add_api_route(
                path.replace("{file_path}", "{file_path:path}"),
                endpoint_for(operation),
                methods=[method.upper()],
            )
    return app
