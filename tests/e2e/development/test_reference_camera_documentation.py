"""Documentation ownership checks for the shared reference-camera contract."""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SHARED_README = REPOSITORY_ROOT / "src/tasks/base/generate_dataset/README.md"
TASK_READMES = (
    REPOSITORY_ROOT / "src/tasks/blcs/README.md",
    REPOSITORY_ROOT / "src/tasks/plcs/README.md",
)

_LEGACY_SCOPE = re.compile(
    r"(?:\blegacy\b|(?<![A-Za-z0-9_])v1(?![A-Za-z0-9_])|time_camera_role_v1|"
    r"従来|旧(?:版|来|系|track-query))",
    re.IGNORECASE,
)
_UNQUALIFIED_FIVE_INPUT_PATTERNS = (
    re.compile(
        r"(?:\ball\b|\bevery\b|全|すべての).{0,80}(?:models?|model)"
        r".{0,240}(?:five|5|五).{0,20}"
        r"(?:public\s+inputs?|inputs?|arguments?|入力|引数|tensors?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:five|5|五).{0,12}(?:public\s+inputs?|公開入力)|"
        r"(?:public\s+inputs?|公開入力).{0,20}(?:five|5|五))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:five|5|五).{0,12}(?:tensors?|inputs?|arguments?|tensor|入力|引数)"
        r".{0,80}(?:model\s*call|forward|signature|contract|公開|契約)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:model\s*call|forward|signature|contract|公開入力|契約)"
        r".{0,160}(?:five|5|五).{0,12}(?:tensors?|inputs?|arguments?|tensor|入力|引数)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:同じ|共通の).{0,20}(?:five|5|五).{0,12}(?:tensors?|inputs?|tensor|入力)"
        r".{0,40}(?:contract|契約)",
        re.IGNORECASE,
    ),
)
_UNQUALIFIED_ROLE_AXIS_PATTERNS = (
    re.compile(
        r"(?:\(\s*time\s*,\s*camera\s*,\s*role\s*\)|role(?:[- ]axis|軸)?)"
        r".{0,160}query\s*(?:=|is|は|を)?\s*0"
        r".{0,160}(?:objects?|players?|groups?|court-player)"
        r".{0,40}(?:=|is|は|を)?\s*1",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\(\s*time\s*,\s*camera\s*,\s*role\s*\)|role(?:[- ]axis|軸)?)"
        r".{0,160}(?:objects?|players?|groups?|court-player)"
        r".{0,40}(?:=|is|は|を)?\s*1"
        r".{0,160}query\s*(?:=|is|は|を)?\s*0",
        re.IGNORECASE,
    ),
)


def _semantic_chunks(text: str) -> tuple[str, ...]:
    return tuple(
        chunk.strip()
        for chunk in re.split(r"(?:[。！？]\s*|(?<=[.!?])\s+|\n+)", text)
        if chunk.strip()
    )


def _find_unqualified_task_contract_claims(text: str) -> tuple[str, ...]:
    claims: list[str] = []
    patterns = _UNQUALIFIED_FIVE_INPUT_PATTERNS + _UNQUALIFIED_ROLE_AXIS_PATTERNS
    for chunk in _semantic_chunks(text):
        if _LEGACY_SCOPE.search(chunk) is not None:
            continue
        if any(pattern.search(chunk) is not None for pattern in patterns):
            claims.append(chunk)
    return tuple(claims)


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
        "rope_dim",
        "lexicographically ordered",
        "`-1` is reserved only for padded",
        "Training chooses the reference",
        "Validation and test use",
        "Direct inference and prediction visualization require",
        "Matching tensor shapes never authorize",
        "fixed_query_track_compressed_v1",
        "pre-promotion checkpoints are rejected",
        "Object UV/visibility",
        "player-local `canonical_pose_3d`",
        "not change. `CourtReferenceFrameProvenance`",
    )
    for term in required_terms:
        assert term in shared, f"shared reference-camera README is missing {term!r}"

    v2_forward_row = re.search(
        r"^\| `time_camera_reference_selector_v1` \|.*$", shared, re.MULTILINE
    )
    assert v2_forward_row is not None
    assert "the same five tensors plus required `reference_view_index: int64[B]`" in (
        v2_forward_row.group(0)
    )
    for coordinate in (
        "query `(t,0,0)`",
        "reference objects `(t,v+1,0)`",
        "other objects `(t,v+1,1)`",
    ):
        assert coordinate in v2_forward_row.group(0)

    for transform in (
        "point_ref   = S_r point_phys",
        "vector_ref  = S_r vector_phys",
        "C_ref       = S_r C_phys",
        "R_cam<-ref  = R_cam<-phys S_r^T",
    ):
        assert transform in shared

    assert "Camera-view v2 datasets and checkpoints are separate artifacts" in shared
    assert "not auto-remapped, dual-written, or upgraded in place" in shared
    assert (
        "Direct inference and prediction visualization require an explicit stable\n"
        "`reference_camera_id`"
    ) in shared


def test_task_readmes_link_to_shared_authority_without_copying_common_formulas() -> (
    None
):
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
            "model=tracking_query_reference",
        ),
        "plcs": (
            "plcs_track_query_reference",
            "human_kp (B,V,T,Q,17,2)",
            "model=tracking_query_reference",
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
        assert _find_unqualified_task_contract_claims(text) == ()


def test_unqualified_task_contract_oracle_catches_equivalent_restatements() -> None:
    contradictory_claims = (
        "All track-query models expose five public inputs.",
        "The interface exposes five public inputs.",
        "Every model uses a five tensor forward signature.",
        "All models share five arguments.",
        "The adapter builds a 5 tensor model call.",
        "single / multiview / track-queryの全modelは公開入力を5 tensorに統一します。",
        "modelの公開入力は5つです。",
        "Track-query variants use the same five inputs under one contract.",
        "各track-query variantは同じ5入力・3出力契約で比較します。",
        "M-RoPE (time,camera,role) assigns query=0 and object=1.",
        "The role axis uses query 0 and player groups 1.",
        "role軸はquery=0、court-player group=1です。",
    )
    for claim in contradictory_claims:
        assert _find_unqualified_task_contract_claims(claim), claim


def test_unqualified_task_contract_oracle_allows_v1_scope_and_observation_shapes() -> (
    None
):
    allowed_task_details = (
        "Legacy v1 uses a five tensor forward signature.",
        "`time_camera_role_v1` track-query variants use the same five inputs under one contract.",
        "v1のrole軸はquery=0、court-player group=1です。",
        "BLCS固有の5観測tensor shapeは ball_uv、ball_vis、court_kp、court_vis、padding_maskです。",
        "PLCS has five observation tensors; v2 also requires its reference field.",
    )
    for detail in allowed_task_details:
        assert _find_unqualified_task_contract_claims(detail) == (), detail
