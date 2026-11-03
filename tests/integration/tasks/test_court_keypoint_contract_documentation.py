"""Documentation authority checks for the shared BLCS/PLCS CourtKP20 contract."""

from __future__ import annotations

from pathlib import Path

from tests.support.documents import assert_local_links, links

_ROOT = Path(__file__).resolve().parents[3]
_SHARED = _ROOT / "src/tasks/base/generate_dataset/README.md"
_TASK_READMES = (
    _ROOT / "src/tasks/blcs/README.md",
    _ROOT / "src/tasks/plcs/README.md",
)


def test_shared_readme_is_the_only_detailed_court_keypoint_authority() -> None:
    shared = _SHARED.read_text(encoding="utf-8")

    for required in (
        "physical_v1",
        "camera_view_v2",
        "physical_courtkp20_v1",
        "camera_view_courtkp20_rzpi_v1",
        "reference_camera_court_rzpi_v1",
        "court_coordinate_normalization",
    ):
        assert required in shared
    assert_local_links(_SHARED, shared)
    for task_readme in _TASK_READMES:
        task_text = task_readme.read_text(encoding="utf-8")
        assert "../base/generate_dataset/README.md" in links(task_text)
        assert_local_links(task_readme, task_text)
    # The permutation and geometry are numerical contracts covered by
    # test_court_keypoint_conventions and test_reference_camera_dataset_roundtrip.
