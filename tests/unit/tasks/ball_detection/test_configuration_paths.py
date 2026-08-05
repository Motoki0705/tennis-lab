"""Ball YouTube archive and role-tagged manual-video path contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.tasks.ball_detection.configuration import BallYoutubePathContract
from src.utils.configuration import ConfigurationError


def _config(root: Path, *, local_video: object) -> object:
    return OmegaConf.create(
        {
            "paths": {
                "project_root": str(root),
                "data_root": "data",
                "checkpoint_root": "ckpt",
                "artifact_root": "artifacts",
                "output_root": "outputs",
                "cache_root": ".cache",
                "external_asset_root": "third_party",
            },
            "workflow": {
                "root": "tennis/dino_ssl",
                "discovery": {"enabled": False},
                "sources": [
                    {
                        "video_id": "manual",
                        "url": "https://example.test/manual",
                        "local_video": local_video,
                    }
                ],
                "download": {"download_archive": "manifests/archive.txt"},
            },
        }
    )


@pytest.mark.parametrize("role", ["data", "external_asset"])
def test_manual_video_resolves_only_from_declared_role(
    tmp_path: Path, role: str
) -> None:
    role_root = tmp_path / ("data" if role == "data" else "third_party")
    video = role_root / "manual/source.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    contract = BallYoutubePathContract.from_config(
        _config(
            tmp_path,
            local_video={"role": role, "path": "manual/source.mp4"},
        )
    )

    assert contract.local_video_for("manual") == video.resolve()
    assert contract.download_archive == (
        tmp_path / "data/tennis/dino_ssl/manifests/archive.txt"
    )


@pytest.mark.parametrize(
    "local_video",
    [
        "/tmp/legacy.mp4",
        {"role": "cache", "path": "manual.mp4"},
        {"role": "data", "path": "../outside.mp4"},
        {"role": "data", "path": "manual.mp4", "typo": True},
    ],
)
def test_manual_video_rejects_legacy_or_invalid_declarations(
    tmp_path: Path, local_video: object
) -> None:
    with pytest.raises((ConfigurationError, TypeError)):
        BallYoutubePathContract.from_config(_config(tmp_path, local_video=local_video))


def test_archive_rejects_absolute_and_escaping_paths(tmp_path: Path) -> None:
    for value in ("/tmp/archive.txt", "../archive.txt"):
        config = _config(tmp_path, local_video=None)
        mutable = OmegaConf.to_container(config, resolve=True)
        assert isinstance(mutable, dict)
        workflow = deepcopy(mutable["workflow"])
        assert isinstance(workflow, dict)
        download = workflow["download"]
        assert isinstance(download, dict)
        download["download_archive"] = value
        mutable["workflow"] = workflow
        with pytest.raises(ConfigurationError):
            BallYoutubePathContract.from_config(OmegaConf.create(mutable))
