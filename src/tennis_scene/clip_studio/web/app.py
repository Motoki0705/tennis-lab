"""Loopback-only HTTP API, ranged video delivery and exact paused-frame previews."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import cv2
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.tennis_scene.clip_studio.export import ExportSettings
from src.tennis_scene.clip_studio.initialization import StartupNotice
from src.tennis_scene.clip_studio.project import ClipStudioProject
from src.tennis_scene.clip_studio.sources import PreviewSource
from src.tennis_scene.clip_studio.timeline import source_frame_index
from src.tennis_scene.clip_studio.web.jobs import JobRequest, Jobs
from src.tennis_scene.clip_studio.web.service import Edit, Editor, RevisionConflict
from src.tennis_scene.configuration import ClipStudioRuntimeConfig

STATIC = Path(__file__).parent / "static"


def create_app(
    runtime: ClipStudioRuntimeConfig,
    project: ClipStudioProject,
    *,
    startup_notice: StartupNotice | None = None,
) -> FastAPI:
    sources: list[PreviewSource] = []
    try:
        for source in project.sources:
            sources.append(
                PreviewSource(
                    source.path,
                    tile_width=runtime.gui.tile_width,
                    cache_frames=runtime.gui.cache_frames,
                    seek_grab_threshold=runtime.gui.seek_grab_threshold,
                )
            )
        editor = Editor(
            project,
            [source.info for source in sources],
            runtime.export.projects_path,
            runtime.export.resolver,
        )
    except Exception:
        for preview in sources:
            preview.close()
        raise
    settings = runtime.export
    jobs = Jobs(
        editor,
        ExportSettings(
            output_dir=settings.output_dir,
            fps=settings.fps,
            width=settings.width,
            height=settings.height,
            crf=settings.crf,
            overwrite=settings.overwrite,
        ),
        runtime.audio_sync,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            jobs.close()
            for source in sources:
                source.close()

    app = FastAPI(title="Tennis Clip Studio", lifespan=lifespan)
    app.state.editor = editor
    app.state.jobs = jobs
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    async def same_origin(
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
                    {"detail": "Cross-origin writes are forbidden"}, status_code=403
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    async def value_error(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            {"detail": str(error)},
            status_code=409 if isinstance(error, RevisionConflict) else 400,
        )

    async def key_error(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=404)

    async def io_error(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            {"detail": f"ファイル操作に失敗しました: {error}"}, status_code=500
        )

    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    def static(name: str) -> FileResponse:
        if name not in {"studio.js", "playback.js", "style.css"}:
            raise HTTPException(404)
        return FileResponse(STATIC / name)

    def get_startup_notice() -> dict[str, Any] | None:
        return asdict(startup_notice) if startup_notice is not None else None

    def get_project() -> dict[str, Any]:
        return editor.snapshot()

    def edit(request: Edit) -> dict[str, Any]:
        return editor.edit(request)

    def media(camera: int) -> FileResponse:
        if not 0 <= camera < len(sources):
            raise HTTPException(404, "Camera not found")
        return FileResponse(sources[camera].video_path)

    def frame(
        camera: int, time: Annotated[float, Query(allow_inf_nan=False)], revision: int
    ) -> Response:
        with editor.lock:
            editor.check_revision(revision)
            if not 0 <= camera < len(sources):
                raise HTTPException(404, "Camera not found")
            info = sources[camera].info
            index = source_frame_index(
                time,
                offset_sec=editor.project.sources[camera].offset_sec,
                fps=info.fps,
                frame_count=info.frame_count,
            )
        if index is None:
            return Response(status_code=204)
        success, data = cv2.imencode(".jpg", sources[camera].get_frame(index))
        if not success:
            raise HTTPException(500, "Preview encoding failed")
        return Response(
            data.tobytes(),
            media_type="image/jpeg",
            headers={"X-Frame-Index": str(index)},
        )

    def start_job(request: JobRequest) -> dict[str, Any]:
        return jobs.start(request)

    def get_job() -> dict[str, Any]:
        return jobs.snapshot()

    def cancel_job() -> dict[str, Any]:
        return jobs.cancel()

    app.middleware("http")(same_origin)
    app.add_exception_handler(ValueError, value_error)
    app.add_exception_handler(KeyError, key_error)
    app.add_exception_handler(OSError, io_error)
    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/static/{name}", static, methods=["GET"])
    app.add_api_route("/api/startup-notice", get_startup_notice, methods=["GET"])
    app.add_api_route("/api/project", get_project, methods=["GET"])
    app.add_api_route("/api/edit", edit, methods=["POST"])
    app.add_api_route("/api/media/{camera}", media, methods=["GET"])
    app.add_api_route("/api/frame/{camera}", frame, methods=["GET"])
    app.add_api_route("/api/jobs", start_job, methods=["POST"])
    app.add_api_route("/api/jobs", get_job, methods=["GET"])
    app.add_api_route("/api/jobs/cancel", cancel_job, methods=["POST"])
    return app
