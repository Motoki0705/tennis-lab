"""FastAPI binding of the shared review app to the PLCS dataset service."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Query
from fastapi.responses import Response

from src.tasks.base.visualization.review.web import create_review_app
from src.tasks.plcs.visualization.review.dataset_service import (
    PLCSDatasetReviewService,
)

TITLE = "PLCS Dataset Review"
STATIC = Path(__file__).parent / "dataset_static"


def create_dataset_app(service: PLCSDatasetReviewService) -> FastAPI:
    app = create_review_app(
        service,
        title=TITLE,
        task="plcs",
        extra_static={
            name: STATIC / name
            for name in ("inspection.css", "inspection.js", "observation.mjs")
        },
        extra_stylesheets=("inspection.css",),
        extra_modules=("inspection.js",),
    )

    @app.get("/api/scene/inspection")
    def inspection(
        form: Annotated[str, Query()],
        scene: Annotated[str, Query()],
        revision: Annotated[str, Query()],
    ) -> dict[str, Any]:
        return service.inspection(form, scene, revision).document

    @app.get("/api/scene/observations")
    def observations(
        form: Annotated[str, Query()],
        scene: Annotated[str, Query()],
        revision: Annotated[str, Query()],
    ) -> Response:
        return Response(
            service.inspection(form, scene, revision).binary,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
        )

    return app


__all__ = ["TITLE", "create_dataset_app"]
