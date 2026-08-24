"""Documentation routing invariants for court-coordinate normalization."""

from __future__ import annotations

import re
from pathlib import Path

from src.utils.paths import PROJECT_ROOT

_CANONICAL_ANCHOR = (
    "src/tasks/base/README.md#court-coordinate-normalization-contract"
)
_CANONICAL_OWNER = Path("src/tasks/base/README.md")
_ROUTED_ENTRY_POINTS = (
    Path("src/tasks/blcs/README.md"),
    Path("src/tasks/plcs/README.md"),
    Path("src/tasks/slcs/README.md"),
    Path("src/tasks/base/configs/court_coordinate_normalization/v1.yaml"),
    Path("src/tasks/base/configs/court_coordinate_normalization/v2.yaml"),
    Path("src/tasks/plcs/configs/court_coordinate_normalization/v1.yaml"),
    Path("src/tasks/plcs/configs/court_coordinate_normalization/v2.yaml"),
    Path("src/tasks/blcs/configs/run/train.yaml"),
    Path("src/tasks/plcs/configs/generate_dataset_norm_v2.yaml"),
    Path("src/tasks/plcs/configs/train_norm_v2.yaml"),
    Path("src/synthetic_data_generation/dataset/plcs/README.md"),
    Path("src/tennis_scene/README.md"),
    Path("src/tennis_scene/generate_dataset/README.md"),
)


def _read(relative_path: Path) -> str:
    text: str = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    return text


def _section(document: str, heading: str) -> str:
    start = document.index(heading)
    match = re.search(r"(?m)^## ", document[start + len(heading) :])
    if match is None:
        return document[start:]
    return document[start : start + len(heading) + match.start()]


def test_base_readme_is_the_complete_shared_contract_owner() -> None:
    document = _read(_CANONICAL_OWNER)
    heading = "## Court-coordinate normalization contract"
    assert document.count(heading) == 1
    contract = _section(document, heading)

    for required_fact in (
        "`v1` | `(5.485, 11.885, 1.07)`",
        "`v2` | `(11.885, 11.885, 11.885)`",
        "position_norm = position_m / scale_xyz",
        "position_m = position_norm * scale_xyz",
        "velocity_norm = velocity_m_per_s / scale_xyz",
        "velocity_m_per_s = velocity_norm * scale_xyz",
        "court_coordinate_normalization=v1",
        "court_coordinate_normalization=v2",
        'position_unit: "m"',
        'velocity_unit: "m/s"',
        "metadata-free",
        "version-qualified",
        "runtime compatibility",
        "materialization.source_normalization_version=v1",
    ):
        assert required_fact in contract

    for responsibility_link in (
        "(../../utils/schema/court_normalization.py)",
        "(data/court_coordinate_contract.py)",
        "(model_io/court_coordinate_contract.py)",
        "(data/court_coordinate_materializer.py)",
        "(configs/materialize_court_coordinate_normalization.yaml)",
    ):
        assert responsibility_link in contract

    for responsibility in (
        "mathematical resolver",
        "dataset metadata schema",
        "checkpoint metadata adapter",
        "materializer",
    ):
        assert responsibility in contract

    assert re.search(r"default.{0,20}`v1`", contract)
    assert re.search(r"`version`.{0,20}`scale_xyz`.{0,30}不一致", contract)
    assert re.search(r"metadata-free.{0,50}`v1` runtime", contract)
    assert "checkpoint weightを自動移行することはありません" in contract


def test_all_documentation_entry_points_route_to_the_canonical_anchor() -> None:
    assert len(_ROUTED_ENTRY_POINTS) == 13
    for relative_path in _ROUTED_ENTRY_POINTS:
        assert _CANONICAL_ANCHOR in _read(relative_path), relative_path


def test_task_local_normalization_facts_remain_with_their_public_interfaces() -> None:
    plcs = _section(
        _read(Path("src/tasks/plcs/README.md")),
        "## Court-coordinate normalization",
    )
    plcs_prose = re.sub(r"\s+", " ", plcs)
    assert re.search(r"`position` translation.{0,40}だけ.{0,40}正規化", plcs_prose)
    for metre_array in ("canonical_pose_3d", "human_kp_3d", "position_court_m"):
        assert metre_array in plcs
    assert "metre のまま" in plcs
    assert "rotation" in plcs and "yaw" in plcs

    slcs = _section(_read(Path("src/tasks/slcs/README.md")), "## 入出力契約")
    for public_fact in (
        "SceneResult",
        "player/ball position",
        "metre値",
        "model境界のposition",
        "position uncertainty",
        "scalar head",
        "mean(scale_xyz)",
    ):
        assert public_fact in slcs

    for relative_path in (
        Path("src/tennis_scene/README.md"),
        Path("src/tennis_scene/generate_dataset/README.md"),
    ):
        scene_result = _read(relative_path)
        for public_fact in (
            "SceneResult",
            "player_position",
            "ball_3d",
            "[m]",
            "provenance",
        ):
            assert public_fact in scene_result, (relative_path, public_fact)
        assert re.search(r"再scaleし(?:ない|ません)", scene_result)

    compact = _read(Path("src/synthetic_data_generation/dataset/plcs/README.md"))
    compact_prose = re.sub(r"\s+", " ", compact)
    for compact_workspace_fact in (
        "supervision.npz",
        "dimensionless PLCS",
        "position_court_m",
        "human_kp_3d",
        "canonical_pose_3d",
        "physical metres",
        "manifest root",
        "logical-scene record",
        "before consuming arrays",
        "1e-5 m",
    ):
        assert compact_workspace_fact in compact_prose


def test_common_contract_prose_is_not_duplicated_outside_the_owner() -> None:
    routed_text = "\n".join(_read(path) for path in _ROUTED_ENTRY_POINTS)

    for owner_only_fact in (
        "position_norm = position_m / scale_xyz",
        "position_norm = position_court_m / scale_xyz",
        "scale_xyz = (5.485, 11.885, 1.07)",
        "(5.485, 11.885, 1.07)m",
        "metadata-free",
        "src/utils/schema/court_normalization.py",
        "src.tasks.base.scripts.materialize_court_coordinate_normalization",
        "materialization.source_normalization_version",
        "Mathematical definition and artifact schema",
    ):
        assert owner_only_fact not in routed_text

    duplicated_policy_patterns = (
        r"(?i)(?:default|既定).{0,40}`?v1`?.{0,40}(?:compat|互換)",
        r"(?i)`?v1`?.{0,40}(?:compat|互換).{0,40}(?:default|既定)",
        r"(?i)metadata.{0,20}(?:のない|を持たない).{0,80}`?v[12]`?",
        r"(?i)(?:missing|unknown|mixed).{0,100}(?:mismatch|不一致|error|例外)",
        r"(?i)(?:norm-v1\|norm-v2|norm_v1.{0,20}norm_v2).{0,100}"
        r"(?:識別|guard|上書き|publication|artifact)",
        r"(?i)(?:artifact|dataset|checkpoint).{0,100}(?:version-qualified|別名)"
        r".{0,100}(?:metadata|識別|上書き)",
    )
    for pattern in duplicated_policy_patterns:
        assert re.search(pattern, routed_text) is None, pattern

    assert re.search(
        r"(?i)(?:mathematical definition|mathematical resolver).{0,80}artifact schema",
        routed_text,
    ) is None
