"""Local HTTP routes exposing only saved data and source RGB."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from src.tasks.player_association.review.service import AssociationReviewService

STATIC = Path(__file__).with_name("static")


def create_app(service: AssociationReviewService) -> FastAPI:
    app = FastAPI(
        title="Player Association dataset review", docs_url=None, redoc_url=None
    )

    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    def static(name: str) -> FileResponse:
        if name not in {"app.js", "style.css"}:
            raise HTTPException(404, "Unknown review asset")
        return FileResponse(STATIC / name)

    def catalog() -> dict[str, Any]:
        return service.catalog()

    def clip(clip: str) -> dict[str, Any]:
        try:
            return service.detail(clip)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    def frame(
        clip: str,
        frame: int,
        report: int | None = None,
        camera: str | None = None,
        track: int | None = None,
    ) -> dict[str, Any]:
        try:
            return service.frame(
                clip, frame, report_id=report, camera=camera, track_id=track
            )
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    def image(clip: str, camera: str, frame: int, track: int | None = None) -> Response:
        try:
            return Response(
                service.image(clip, camera, frame, track),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/static/{name}", static, methods=["GET"])
    app.add_api_route("/api/catalog", catalog, methods=["GET"])
    app.add_api_route("/api/clip", clip, methods=["GET"])
    app.add_api_route("/api/frame", frame, methods=["GET"])
    app.add_api_route("/api/image", image, methods=["GET"])
    return app
