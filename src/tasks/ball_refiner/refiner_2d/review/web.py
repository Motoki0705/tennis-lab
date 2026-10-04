"""Small task-local GET-only review server; no generation or model execution."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.tasks.ball_refiner.refiner_2d.review.artifacts import ReviewArtifacts


def create_app(artifacts: ReviewArtifacts) -> FastAPI:
    app = FastAPI(title="Ball Refiner 2D · Dataset review")
    static = Path(__file__).with_name("static")
    lock = RLock()
    app.mount("/static", StaticFiles(directory=static), name="static")

    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    def catalog() -> dict[str, Any]:
        return artifacts.catalog()

    def timeline(clip: str) -> dict[str, Any]:
        try:
            with lock:
                return artifacts.timeline(clip)
        except KeyError as error:
            raise HTTPException(404, "Clip not found") from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(422, str(error)) from error

    def frame(clip: str, frame: int, condition: str = "observed") -> dict[str, Any]:
        try:
            with lock:
                return artifacts.frame(clip, frame, condition)
        except KeyError as error:
            raise HTTPException(404, "Clip or artifact field not found") from error
        except IndexError as error:
            raise HTTPException(422, str(error)) from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(422, str(error)) from error

    def image(clip: str, frame: int) -> Response:
        try:
            with lock:
                if not 0 <= frame < artifacts.clips[clip].frame_count:
                    raise IndexError("Frame outside timeline")
                payload = artifacts.image(clip, frame)
                return Response(payload, media_type="image/jpeg")
        except (KeyError, FileNotFoundError) as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, IndexError) as error:
            raise HTTPException(422, str(error)) from error

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/api/catalog", catalog, methods=["GET"])
    app.add_api_route("/api/timeline", timeline, methods=["GET"])
    app.add_api_route("/api/frame", frame, methods=["GET"])
    app.add_api_route("/api/image", image, methods=["GET"])
    return app
