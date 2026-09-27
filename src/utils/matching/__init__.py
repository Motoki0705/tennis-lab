"""Assignment and clustering solvers shared across tasks."""

from src.utils.matching.multiview_clustering import (
    MultiviewClustering,
    cluster_multiview,
    decision_margins,
)

__all__ = ["MultiviewClustering", "cluster_multiview", "decision_margins"]
