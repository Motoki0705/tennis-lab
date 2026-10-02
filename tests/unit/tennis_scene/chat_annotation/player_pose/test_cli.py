"""The public CLI validates caller-owned paths before creating artifacts."""

from pathlib import Path

import pytest

from src.tennis_scene.chat_annotation.player_pose import __main__ as cli
from src.utils.configuration.errors import PathContractError


def test_relative_campaign_is_rejected_before_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setattr(
        cli, "initialize", lambda *_: pytest.fail("initialization must not run")
    )
    with pytest.raises(PathContractError, match="absolute"):
        cli.main(["init", "--campaign", "relative", "--config", str(config)])


def test_missing_config_does_not_create_campaign(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    with pytest.raises(PathContractError):
        cli.main(
            [
                "init",
                "--campaign",
                str(campaign),
                "--config",
                str(tmp_path / "missing.json"),
            ]
        )
    assert not campaign.exists()


@pytest.mark.parametrize("index", ["-1", "1"])
def test_unusable_index_is_rejected_before_status_side_effects(
    tmp_path: Path, index: str
) -> None:
    with pytest.raises(SystemExit):
        cli.main(["status", "--campaign", str(tmp_path), "--index", index])
