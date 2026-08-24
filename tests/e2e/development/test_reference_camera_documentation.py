"""Documentation ownership checks for the shared reference-camera contract."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SHARED_README = REPOSITORY_ROOT / "src/tasks/base/generate_dataset/README.md"
TASK_READMES = (
    REPOSITORY_ROOT / "src/tasks/blcs/README.md",
    REPOSITORY_ROOT / "src/tasks/plcs/README.md",
)


def test_shared_readme_owns_reference_camera_artifact_and_runtime_semantics() -> None:
    shared = SHARED_README.read_text(encoding="utf-8")
    required_terms = (
        "physical_courtkp20_v1",
        "camera_view_courtkp20_rzpi_v1",
        "reference_camera_court_rzpi_v1",
        "time_camera_role_v1",
        "time_camera_reference_selector_v1",
        "reference_view_index",
        "view_camera_ids",
        "reference_camera_id",
        "reference_from_physical",
        "physical_from_reference",
        "query `(t,0,0)`",
        "reference objects `(t,v+1,0)`",
        "other objects `(t,v+1,1)`",
        "required `reference_view_index: int64[B]`",
        "selector_zero",
        "role_rope_enabled",
        "rope_dim",
        "lexicographically ordered",
        "`-1` is reserved only for padded",
        "Training chooses the reference",
        "Validation and test use",
        "Direct inference and prediction visualization require",
        "Matching tensor shapes never authorize",
        "Object UV/visibility",
        "player-local `canonical_pose_3d`",
        "not change. `CourtReferenceFrameProvenance`",
    )
    for term in required_terms:
        assert term in shared, f"shared reference-camera README is missing {term!r}"


def test_task_readmes_link_to_shared_authority_without_copying_common_formulas() -> None:
    prohibited_duplicates = (
        "point_ref   = S_r point_phys",
        "vector_ref  = S_r vector_phys",
        "R_cam<-ref  = R_cam<-phys S_r^T",
        "query `(t,0,0)`; reference objects `(t,v+1,0)`",
        "lexicographically ordered scene ID table",
    )
    task_specific_terms = {
        "blcs": (
            "blcs_track_query_reference",
            "ball_uv (B,V,T,Q,2)",
            "track_query_ablation_d_v2_selector_zero",
        ),
        "plcs": (
            "plcs_track_query_reference",
            "human_kp (B,V,T,Q,17,2)",
            "track_query_ablation_d_v2_selector_zero",
        ),
    }
    for readme in TASK_READMES:
        text = readme.read_text(encoding="utf-8")
        assert "../base/generate_dataset/README.md" in text
        for duplicate in prohibited_duplicates:
            assert duplicate not in text
        task = readme.parent.name
        for term in task_specific_terms[task]:
            assert term in text
