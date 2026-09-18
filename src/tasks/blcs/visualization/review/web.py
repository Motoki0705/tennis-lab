"""FastAPI binding of the shared review app to the BLCS dataset service."""

from __future__ import annotations

from fastapi import FastAPI

from src.tasks.base.visualization.review.web import create_review_app
from src.tasks.blcs.visualization.review.dataset_service import (
    BLCSDatasetReviewService,
)

TITLE = "BLCS Dataset Review"


def create_dataset_app(service: BLCSDatasetReviewService) -> FastAPI:
    return create_review_app(service, title=TITLE, task="blcs")


__all__ = ["TITLE", "create_dataset_app"]
