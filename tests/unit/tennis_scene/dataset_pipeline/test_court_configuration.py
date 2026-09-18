"""Court refinement is opt-in and configuration errors fail explicitly."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.tennis_scene.dataset_pipeline.configuration import DatasetBuildConfig


@pytest.fixture
def court_config(monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(
        config_dir=str(root / "src/tennis_scene/configs"), version_base="1.3"
    ):
        cfg = compose(config_name="build_broadcast_slcs_dataset")
    cfg.court_calibration_clips = None
    cfg.excluded_clips = {}
    cfg.dataset_clip_ids = None
    cfg.clip_ids = None
    monkeypatch.setattr(
        "src.tennis_scene.dataset_pipeline.configuration.load_dataset_manifest",
        lambda path: SimpleNamespace(
            clips={"video/clip": SimpleNamespace(video_id="video")}
        ),
    )
    return cfg


@pytest.mark.parametrize("value", [None, 0, 20.0, 0.5])
def test_court_refinement_padding_accepts_explicit_values(
    court_config: DictConfig, value: float | None
) -> None:
    with open_dict(court_config.court):
        court_config.court.crop_refinement_padding_px = value
    assert (
        DatasetBuildConfig.from_config(court_config).court.crop_refinement_padding_px
        == value
    )


def test_absent_refinement_keeps_legacy_config_identity(
    court_config: DictConfig,
) -> None:
    before = OmegaConf.to_container(court_config.court, resolve=True)
    assert "crop_refinement_padding_px" not in court_config.court
    assert (
        DatasetBuildConfig.from_config(court_config).court.crop_refinement_padding_px
        is None
    )
    assert OmegaConf.to_container(court_config.court, resolve=True) == before


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "20"])
def test_court_refinement_padding_rejects_invalid_values(
    court_config: DictConfig, value: object
) -> None:
    with open_dict(court_config.court):
        court_config.court.crop_refinement_padding_px = value
    with pytest.raises(ValueError, match="crop_refinement_padding_px"):
        DatasetBuildConfig.from_config(court_config)


def test_unknown_court_refinement_key_is_rejected(court_config: DictConfig) -> None:
    with open_dict(court_config.court):
        court_config.court.crop_refinement_padding = 20
    with pytest.raises(ValueError, match="crop_refinement_padding"):
        DatasetBuildConfig.from_config(court_config)
