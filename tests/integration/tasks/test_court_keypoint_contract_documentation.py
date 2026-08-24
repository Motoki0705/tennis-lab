"""Documentation authority checks for the shared BLCS/PLCS CourtKP20 contract."""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SHARED = _ROOT / "src/tasks/base/generate_dataset/README.md"
_TASK_READMES = (
    _ROOT / "src/tasks/blcs/README.md",
    _ROOT / "src/tasks/plcs/README.md",
)


def test_shared_readme_is_the_only_detailed_court_keypoint_authority() -> None:
    shared = _SHARED.read_text(encoding="utf-8")
    compact_shared = re.sub(r"\s+", "", shared)

    for required in (
        "physical_v1",
        "camera_view_v2",
        "physical_courtkp20_v1",
        "camera_view_courtkp20_rzpi_v1",
        "Camera-local disk semantics",
        "Model reference semantics",
        "H_v^-1 o H_r",
        "#782/#788",
        "court_coordinate_normalization",
        "separate metadata field",
        "version axis",
        "dataset regeneration",
        "model retraining",
    ):
        assert required in shared
    assert (
        "(3,2,1,0,7,6,5,4,11,10,9,8,13,12,14,17,18,15,16,19)"
        in compact_shared
    )

    authority_only_fragments = (
        "physical_courtkp20_v1",
        "camera_view_courtkp20_rzpi_v1",
        "H_v^-1 o H_r",
        "OPPOSITE_COURT_END_INDEX",
        "(3,2,1,0,7,6,5,4,11,10,9,8,13,12,14,17,18,15,16,19)",
    )
    for task_readme in _TASK_READMES:
        task_text = task_readme.read_text(encoding="utf-8")
        compact_task = re.sub(r"\s+", "", task_text)
        assert task_text.count("../base/generate_dataset/README.md") == 1
        for fragment in authority_only_fragments:
            assert fragment not in task_text
            assert fragment not in compact_task
