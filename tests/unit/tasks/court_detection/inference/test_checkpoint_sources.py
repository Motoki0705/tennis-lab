"""A trained Court model and its pretrained backbone keep separate authorities."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import src.tasks.court_detection.inference.predictor as predictor_module
import src.tasks.court_detection.visualization.api.predict as visualization_api
from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots


class _SourceProbe(CourtPredictor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The loader boundary is real; neural construction belongs to loader tests.
        pass


def _resolver(root: Path) -> PathResolver:
    return PathResolver(RuntimePathRoots(
        project_root=root,
        data_root=root / "data",
        checkpoint_root=root / "ckpt",
        artifact_root=root / "outputs",
        output_root=root / "outputs",
        cache_root=root / "cache",
        external_asset_root=root / "third_party",
    ))


@pytest.mark.parametrize("head", ["kp", "seg", "line", "semantic_line"])
def test_visualization_loads_declared_artifact_without_rebasing_pretrained_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head: Any,
) -> None:
    resolver = _resolver(tmp_path)
    pretrained = tmp_path / "ckpt/dinov3/pretrained.pth"
    pretrained.parent.mkdir(parents=True)
    pretrained.write_bytes(b"pretrained")
    (tmp_path / "ckpt/model.ckpt").write_bytes(b"different-model")
    artifact = tmp_path / "outputs/model.ckpt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"trained-model")
    loaded: list[Path] = []

    def load(path: Path, **kwargs: Any) -> Any:
        assert path.read_bytes() == b"trained-model"
        assert kwargs["resolver"] is resolver
        assert resolver.resolve(PathRole.CHECKPOINT, "dinov3/pretrained.pth").read_bytes() == b"pretrained"
        loaded.append(path)
        return SimpleNamespace(model_io=object(), identity={})

    monkeypatch.setattr(predictor_module, "load_court_checkpoint", load)
    for name in ("CourtKeypointPredictor", "CourtSegPredictor", "CourtLinePredictor", "CourtSemanticLinePredictor"):
        monkeypatch.setattr(visualization_api, name, _SourceProbe)
    visualization_api.build_court_visualization_pipeline(
        head, checkpoint_path="model.ckpt", checkpoint_role=PathRole.ARTIFACT,
        resolver=resolver, device="cpu",
    )
    assert loaded == [artifact]
    assert resolver.roots.checkpoint_root == tmp_path / "ckpt"


@pytest.mark.parametrize("case", ["missing", "outside", "unsupported", "no_resolver"])
def test_invalid_artifact_input_stops_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    resolver = _resolver(tmp_path)
    decoy = tmp_path / "ckpt/model.ckpt"
    decoy.parent.mkdir()
    decoy.write_bytes(b"must-not-be-selected")
    load = Mock()
    monkeypatch.setattr(predictor_module, "load_court_checkpoint", load)
    candidate: str | Path = tmp_path / "foreign.ckpt" if case == "outside" else "model.ckpt"
    with pytest.raises((ValueError, FileNotFoundError)):
        _SourceProbe.load_from_checkpoint(
            candidate,
            device="cpu",
            resolver=None if case == "no_resolver" else resolver,
            checkpoint_role=PathRole.DATA if case == "unsupported" else PathRole.ARTIFACT,
        )
    load.assert_not_called()
