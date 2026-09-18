"""FastAPI binding of the shared review app to the PLCS dataset service."""

from __future__ import annotations

from fastapi import FastAPI

from src.tasks.base.visualization.review.web import create_review_app
from src.tasks.plcs.visualization.review.dataset_service import (
    PLCSDatasetReviewService,
)

TITLE = "PLCS Dataset Review"


def create_dataset_app(service: PLCSDatasetReviewService) -> FastAPI:
    return create_review_app(service, title=TITLE, task="plcs")


__all__ = ["TITLE", "create_dataset_app"]
