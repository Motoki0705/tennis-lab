"""FastAPI binding of the shared review app to the SLCS dataset service."""

from __future__ import annotations

from fastapi import FastAPI

from src.tasks.base.visualization.review.web import create_review_app
from src.tasks.slcs.visualization.review.dataset_service import (
    SLCSDatasetReviewService,
)

TITLE = "SLCS Dataset Review"


def create_dataset_app(service: SLCSDatasetReviewService) -> FastAPI:
    return create_review_app(service, title=TITLE, task="slcs")


__all__ = ["TITLE", "create_dataset_app"]
