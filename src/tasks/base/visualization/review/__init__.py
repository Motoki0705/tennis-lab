"""Shared, read-only dataset scene review core for BLCS and PLCS.

The package provides a task-agnostic browser viewer: a catalog over
``data/<task>/<form>/scenes``, camera/frustum parsing, court geometry, and a
FastAPI app that streams scene metadata plus a compact float32 entity binary.
Task packages (``src.tasks.{blcs,plcs}.visualization.review``) bind their own
service to :func:`web.create_review_app`.
"""

from src.tasks.base.visualization.review.camera import (
    FRUSTUM_EDGES,
    CameraRecord,
    parse_cameras,
)
from src.tasks.base.visualization.review.catalog import (
    DatasetCatalog,
    DatasetCatalogError,
    DatasetForm,
)
from src.tasks.base.visualization.review.court import (
    apron_polygon,
    court_edges,
    court_keypoints,
    net_geometry,
)
from src.tasks.base.visualization.review.payload import (
    ScenePayload,
    pack_entity_frames,
)
from src.tasks.base.visualization.review.service import (
    DatasetSceneReviewService,
    SceneArrays,
    load_json_object,
)

__all__ = [
    "FRUSTUM_EDGES",
    "CameraRecord",
    "DatasetCatalog",
    "DatasetCatalogError",
    "DatasetForm",
    "DatasetSceneReviewService",
    "SceneArrays",
    "ScenePayload",
    "apron_polygon",
    "court_edges",
    "court_keypoints",
    "load_json_object",
    "net_geometry",
    "pack_entity_frames",
    "parse_cameras",
]
