"""FastAPI binding of the shared review app to the SLCS dataset service."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Query
from fastapi.responses import Response

from src.tasks.base.visualization.review.web import create_review_app
from src.tasks.slcs.visualization.review.dataset_service import (
    SLCSDatasetReviewService,
)

TITLE = "SLCS Dataset Review"


def create_dataset_app(service: SLCSDatasetReviewService) -> FastAPI:
    static = Path(__file__).parent / "static"
    assets = ("slcs-inspection.css", "slcs-inspection.mjs", "slcs-overlay.mjs", "slcs-framing.mjs")
    app = create_review_app(
        service,
        title=TITLE,
        task="slcs",
        extra_static={name: static / name for name in assets},
        extra_stylesheets=("slcs-inspection.css",),
        extra_modules=("slcs-inspection.mjs",),
    )

    def frame(
        form: Annotated[str, Query()],
        scene: Annotated[str, Query()],
        camera: Annotated[str, Query()],
        frame: Annotated[int, Query()],
        revision: Annotated[str, Query()],
    ) -> dict[str, Any]:
        return service.inspection_frame(form, scene, camera, frame, revision)

    def image(
        form: Annotated[str, Query()],
        scene: Annotated[str, Query()],
        camera: Annotated[str, Query()],
        frame: Annotated[int, Query()],
        revision: Annotated[str, Query()],
    ) -> Response:
        return Response(
            service.image(form, scene, camera, frame, revision),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-SLCS-Frame": str(frame),
            },
        )

    app.add_api_route("/api/inspection/frame", frame, methods=["GET"])
    app.add_api_route("/api/inspection/image", image, methods=["GET"])
    return app


__all__ = ["TITLE", "create_dataset_app"]
