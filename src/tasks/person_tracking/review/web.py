"""Local read-only HTTP routes; no mutation, inference or filesystem browser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.utils.checksum import FileIntegrityError

from .model import ReviewSequence
from .service import TrackingReviewService


def create_app(service: TrackingReviewService) -> FastAPI:
    app = FastAPI(title="Person Tracking dataset review", docs_url=None, redoc_url=None)
    static = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    def sequence(key: str) -> ReviewSequence:
        try:
            return service.load(key)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, OSError, FileIntegrityError) as error:
            raise HTTPException(409, str(error)) from error

    def index() -> FileResponse:
        return FileResponse(
            static / "index.html", headers={"Cache-Control": "no-store"}
        )

    def catalog() -> dict[str, Any]:
        return service.catalog()

    def summary(key: str) -> dict[str, Any]:
        return sequence(key).summary()

    def frame_info(key: str, frame: int) -> dict[str, Any]:
        try:
            return sequence(key).frame_info(frame)
        except IndexError as error:
            raise HTTPException(404, str(error)) from error

    def image(key: str, frame: int) -> Response:
        item = sequence(key)
        if not 0 <= frame < item.frame_count:
            raise HTTPException(404, "Frame outside the exact source timeline")
        try:
            pixels = item.images.read(frame)
            if pixels.shape != (item.height, item.width, 3):
                raise ValueError(
                    "Decoded image dimensions differ from the tracking source"
                )
            okay, data = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not okay:
                raise ValueError("Failed to encode the decoded source frame")
        except (ValueError, OSError, IndexError, FileIntegrityError) as error:
            raise HTTPException(409, str(error)) from error
        return Response(
            data.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/api/catalog", catalog, methods=["GET"])
    app.add_api_route("/api/sequences/{key}", summary, methods=["GET"])
    app.add_api_route(
        "/api/sequences/{key}/frames/{frame}", frame_info, methods=["GET"]
    )
    app.add_api_route(
        "/api/sequences/{key}/frames/{frame}/image", image, methods=["GET"]
    )
    return app
