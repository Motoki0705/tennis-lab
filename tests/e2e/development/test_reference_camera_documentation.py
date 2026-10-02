"""Shared camera geometry stays authoritative after axial-only task cleanup."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SHARED_README = REPOSITORY_ROOT / "src/tasks/base/generate_dataset/README.md"
TASK_READMES = (
    REPOSITORY_ROOT / "src/tasks/blcs/README.md",
    REPOSITORY_ROOT / "src/tasks/plcs/README.md",
)
REFERENCE_TRANSFORMS = (
    "point_ref   = S_r point_phys",
    "vector_ref  = S_r vector_phys",
    "C_ref       = S_r C_phys",
    "R_cam<-ref  = R_cam<-phys S_r^T",
)


def test_shared_readme_owns_camera_geometry_and_artifact_semantics() -> None:
    shared = SHARED_README.read_text(encoding="utf-8")
    for term in (
        "physical_courtkp20_v1",
        "camera_view_courtkp20_rzpi_v1",
        "reference_camera_court_rzpi_v1",
        "stable camera ID",
        "CourtReferenceFrameProvenance",
        "build_reference_frame_provenance()",
        "Object UV/visibility",
        "player-local `canonical_pose_3d`",
        "Camera-view v2 datasets and checkpoints are separate artifacts",
        "not auto-remapped, dual-written, or upgraded in place",
        *REFERENCE_TRANSFORMS,
    ):
        assert term in shared, f"shared camera-geometry README is missing {term!r}"

    assert "Standalone BLCS/PLCS generation, training, and inference accept only" in shared
    assert "`court_keypoints=physical_v1`" in shared
    assert "tennis-scene camera-geometry pipeline" in shared


def test_task_readmes_link_to_shared_geometry_and_describe_supported_model() -> None:
    for readme in TASK_READMES:
        text = readme.read_text(encoding="utf-8")
        task = readme.parent.name
        assert "../base/generate_dataset/README.md" in text
        assert f"models/{task}_multiview_axial_model.py" in text
        assert f"data/{task}/single_object" in text
        assert "`physical_v1` のみ" in text
        assert "model=tracking_query_reference" not in text
        for transform in REFERENCE_TRANSFORMS:
            assert transform not in text
