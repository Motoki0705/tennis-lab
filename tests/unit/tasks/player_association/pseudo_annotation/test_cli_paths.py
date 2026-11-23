from pathlib import Path

import pytest

from src.tasks.player_association.pseudo_annotation.__main__ import main
from src.utils.configuration import PathContractError


def test_prepare_rejects_wrong_plan_kind_before_opening_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("snapshot").mkdir()
    Path("plan-directory").mkdir()

    with pytest.raises(
        PathContractError, match=r"pseudo_annotation\.plan must be an existing file"
    ):
        main(
            [
                "prepare",
                "snapshot",
                "plan-directory",
                "video_000/clip_032",
                "packets/target",
            ]
        )

    assert not Path("packets").exists()
