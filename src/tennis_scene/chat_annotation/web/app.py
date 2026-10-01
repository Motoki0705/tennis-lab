"""Local read-only progress, delegation and ephemeral video review server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..runtime.contracts import BallAnnotation, PlayerAnnotation, SupportedAnnotation
from .catalog import TARGETS, Catalog, RevisionConflict
from .handoff import make_handoff
from .metrics import missing_summary
from .preview import PreviewBusy, PreviewCancelled, PreviewRenderer, combine

STATIC = Path(__file__).parent / "static"


def _default_targets() -> list[Literal["ball", "player"]]:
    return ["ball", "player"]


class VersionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ball: str | None = None
    player: str | None = None


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str
    versions: VersionSelection
    target: Literal["both", "ball", "player"] = "both"


class HandoffItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_id: str
    targets: list[Literal["ball", "player"]] = Field(
        default_factory=_default_targets, min_length=1, max_length=2
    )
    versions: VersionSelection | None = None
    refine: bool = False


class HandoffRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str
    view: Literal["working", "accepted"] = "working"
    selections: list[HandoffItem] = Field(min_length=1, max_length=20)


def create_app(root: Path, *, renderer: PreviewRenderer | None = None) -> FastAPI:
    catalog = Catalog(root)
    previews = renderer or PreviewRenderer()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            previews.close()

    app = FastAPI(title="Annotation Desk", lifespan=lifespan)
    app.state.catalog = catalog
    app.state.previews = previews
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def local_headers(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        origin = request.headers.get("origin")
        if request.method != "GET" and origin:
            parsed = urlsplit(origin)
            if (
                parsed.netloc != request.headers.get("host")
                or parsed.scheme != request.url.scheme
            ):
                return JSONResponse(
                    {"detail": "Cross-origin requests are forbidden"}, status_code=403
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ValueError)
    async def bad_value(request: Request, error: ValueError) -> JSONResponse:
        code = (
            409
            if isinstance(error, RevisionConflict)
            else 429
            if isinstance(error, PreviewBusy)
            else 422
        )
        return JSONResponse({"detail": str(error)}, status_code=code)

    @app.exception_handler(KeyError)
    async def not_found(request: Request, error: KeyError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        if name not in {"style.css", "app.js"}:
            raise HTTPException(404)
        return FileResponse(STATIC / name)

    @app.get("/api/catalog")
    def inventory(view: Literal["working", "accepted"] = "working") -> dict[str, Any]:
        return catalog.snapshot(view)

    @app.post("/api/refresh")
    def refresh(view: Literal["working", "accepted"] = "working") -> dict[str, Any]:
        catalog.refresh()
        return catalog.snapshot(view)

    @app.get("/api/clips/{clip_id}")
    def detail(
        clip_id: str, revision: str, view: Literal["working", "accepted"] = "working"
    ) -> dict[str, Any]:
        with catalog.lock:
            catalog.check_revision(revision)
            clip = catalog.clips[clip_id]
            catalog.check_clip_current(clip)
            return {
                "id": clip_id,
                "revision": catalog.revision,
                "manifest": catalog.relative(clip.manifest_path),
                "manifest_sha256": clip.manifest_sha,
                "video_path": str(clip.video) if clip.video else None,
                "video_sha256": clip.manifest.sha256,
                "times": [
                    float(f.clip_pts * Fraction(clip.manifest.time_base))
                    for f in clip.manifest.frames
                ],
                "source_frames": [f.source_frame_index for f in clip.manifest.frames],
                "is_target": [f.is_target for f in clip.manifest.frames],
                "targets": {
                    t: {
                        "default_version": (
                            selected.id
                            if (selected := catalog.default_version(clip, t, view))
                            else None
                        ),
                        "versions": [
                            catalog.version_payload(v) for v in clip.versions[t]
                        ],
                        "missing": missing_summary(len(clip.manifest.frames)),
                    }
                    for t in TARGETS
                },
                "warnings": clip.warnings,
                "errors": clip.errors,
            }

    @app.get("/api/clips/{clip_id}/annotations/{target}/{version_id}")
    def annotation(
        clip_id: str,
        target: str,
        version_id: str,
        revision: str,
        download: bool = False,
    ) -> Response:
        with catalog.lock:
            catalog.check_revision(revision)
            version = catalog.select(clip_id, target, version_id)
            assert version is not None
            catalog.check_clip_current(catalog.clips[clip_id])
            data, stats, timeline = catalog.read_version(version)
            if download:
                return Response(
                    data.model_dump_json(indent=2),
                    media_type="application/json",
                    headers={
                        "Content-Disposition": f'attachment; filename="{target}_{clip_id}.json"'
                    },
                )
            return JSONResponse(
                {
                    "version": catalog.version_payload(version),
                    "statistics": stats,
                    "timeline": timeline,
                    "annotation": data.model_dump(mode="json"),
                }
            )

    @app.get("/api/clips/{clip_id}/video")
    def video(clip_id: str, revision: str, download: bool = False) -> FileResponse:
        with catalog.lock:
            catalog.check_revision(revision)
            clip = catalog.clips[clip_id]
            catalog.check_clip_current(clip)
            if clip.video is None or clip.errors:
                raise HTTPException(409, "元動画を利用できません")
            path = catalog.safe(clip.video)
            if not path.is_file():
                raise HTTPException(409, "元動画が変更されています。再読込してください")
            return FileResponse(
                path, media_type="video/mp4", filename=path.name if download else None
            )

    @app.post("/api/clips/{clip_id}/preview")
    async def preview(clip_id: str, selection: Selection, request: Request) -> Response:
        with catalog.lock:
            catalog.check_revision(selection.revision)
            clip = catalog.clips[clip_id]
            catalog.check_clip_current(clip)
            if clip.video is None or clip.errors:
                raise HTTPException(409, "動画/manifestのエラーを確認してください")
            data: dict[str, SupportedAnnotation | None] = {}
            versions = selection.versions.model_dump()
            for target in TARGETS:
                version = catalog.select(clip_id, target, versions.get(target))
                if selection.target not in {"both", target}:
                    data[target] = None
                    continue
                if version is None:
                    data[target] = None
                else:
                    value, stats, _ = catalog.read_version(version)
                    if stats["state"] == "invalid":
                        raise HTTPException(
                            422, f"{target}: JSONの検証エラーを解消してください"
                        )
                    data[target] = value
            if not any(data.values()):
                raise HTTPException(
                    422, "選択対象の注釈JSONがありません。元動画で確認してください"
                )
            ball, player = data["ball"], data["player"]
            assert ball is None or isinstance(ball, BallAnnotation)
            assert player is None or isinstance(player, PlayerAnnotation)
            overlay = (
                combine(clip.manifest, ball, player)
                if selection.target == "both"
                else data[selection.target]
            )
            assert overlay is not None
            video_path, manifest = catalog.safe(clip.video), clip.manifest
        try:
            content = await previews.render(
                video_path, manifest, overlay, request.is_disconnected
            )
        except PreviewCancelled as error:
            raise HTTPException(499, str(error)) from error
        except (OSError, RuntimeError) as error:
            raise HTTPException(
                422, f"プレビュー生成に失敗しました: {error}"
            ) from error
        return Response(
            content,
            media_type="video/mp4",
            headers={
                "Cache-Control": "no-store",
                "X-Preview-Storage": "memory-only",
                "Content-Disposition": "inline",
                "X-Catalog-Revision": selection.revision,
            },
        )

    @app.post("/api/handoff")
    def handoff(value: HandoffRequest) -> dict[str, Any]:
        with catalog.lock:
            return make_handoff(
                catalog,
                value.revision,
                [item.model_dump(exclude_none=True) for item in value.selections],
                value.view,
            )

    return app
