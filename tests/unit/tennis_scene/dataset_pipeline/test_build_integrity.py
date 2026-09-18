"""Integrity failures stop the build while ordinary clip failures remain local."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.tennis_scene.dataset_pipeline import build
from src.utils.checksum import FileIntegrityError


@pytest.mark.parametrize("integrity", [True, False])
@pytest.mark.parametrize("preflight_mismatch", [True, False])
def test_build_records_failure_and_stops_only_for_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    integrity: bool,
    preflight_mismatch: bool,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint")
    runtime = SimpleNamespace(
        stage="all",
        seed=42,
        output=tmp_path / "output",
        source=tmp_path,
        destination=tmp_path / "destination",
        clip_ids=("first", "second"),
        dataset_clip_ids=None,
        features_enabled=True,
        feature_checkpoint=checkpoint,
        court=SimpleNamespace(checkpoint=checkpoint),
        checkpoint_sha256={
            name: "0" * 64
            for name in ("plcs", "blcs", "dino", "vitpose", "court", "dinov3")
        }
        if preflight_mismatch
        else None,
    )
    paths = SimpleNamespace(
        plcs_checkpoint=checkpoint,
        blcs_checkpoint=checkpoint,
        dino_checkpoint=checkpoint,
        vitpose_checkpoint=checkpoint,
    )
    monkeypatch.setattr(build.DatasetBuildConfig, "from_config", lambda cfg: runtime)
    monkeypatch.setattr(build.ReferenceClipPaths, "from_config", lambda cfg: paths)
    monkeypatch.setattr(build, "resolved_recipe", lambda cfg, runtime: {"stage": "all"})
    monkeypatch.setattr(
        build,
        "load_dataset_manifest",
        lambda path: SimpleNamespace(
            clips={name: SimpleNamespace(path=name) for name in runtime.clip_ids}
        ),
    )
    monkeypatch.setattr(build.ClipManifest, "load", lambda path: path)
    monkeypatch.setattr(build, "materialize_dataset", MagicMock())
    monkeypatch.setattr(build.torch.cuda, "empty_cache", MagicMock())
    features = MagicMock()
    monkeypatch.setattr(build, "_precompute_features", features)
    error = (
        FileIntegrityError("providers disagree", details={"path": "checkpoint"})
        if integrity
        else ValueError("quality failure")
    )
    process = MagicMock(side_effect=[error, None])
    monkeypatch.setattr(build, "_process_clip", process)
    if preflight_mismatch:
        with pytest.raises(RuntimeError, match="checkpoint"):
            build.build_dataset(OmegaConf.create({}))
        process.assert_not_called()
        features.assert_not_called()
        assert not runtime.output.exists()
        return
    with pytest.raises(FileIntegrityError if integrity else RuntimeError) as caught:
        build.build_dataset(OmegaConf.create({}))
    if integrity:
        assert caught.value is error
    assert process.call_count == (1 if integrity else 2)
    features.assert_not_called()
    failure = json.loads((runtime.output / "failures.json").read_text())
    assert failure == {
        "stage": "all",
        "failures": {"first": f"{type(error).__name__}: {error}"},
    }
