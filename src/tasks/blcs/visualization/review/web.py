"""FastAPI binding of the shared review app to the BLCS dataset service."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Query

from src.tasks.base.visualization.review.web import create_review_app
from src.tasks.blcs.visualization.review.dataset_service import (
    BLCSDatasetReviewService,
)

TITLE = "BLCS Dataset Review"
STATIC = Path(__file__).parent / "static"


def create_dataset_app(service: BLCSDatasetReviewService) -> FastAPI:
    app = create_review_app(
        service,
        title=TITLE,
        task="blcs",
        extra_static={
            name: STATIC / name
            for name in (
                "blcs-inspection.css",
                "blcs-inspection.mjs",
                "blcs-observation.mjs",
            )
        },
        extra_stylesheets=("blcs-inspection.css",),
        extra_modules=("blcs-inspection.mjs",),
    )

    @app.get("/api/inspection")
    def inspection(
        form: Annotated[str, Query()],
        scene: Annotated[str, Query()],
        revision: Annotated[str, Query()],
    ) -> dict[str, Any]:
        return service.inspection(form, scene, revision)

    return app


__all__ = ["TITLE", "create_dataset_app"]
