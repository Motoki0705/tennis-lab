"""D: logical input contracts across declared E: aliases, via real consumers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.tennis_scene.clip_studio.layout import discover_clip_studio_layout
from src.tennis_scene.clip_studio.project import ClipSource
from src.utils.configuration import (
    PathContractError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    local_storage,
)
from tests.support.synthetic_data_generation.configuration import compose_scene_config


@pytest.fixture
def tiered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, mount = tmp_path / "D/repo", tmp_path / "E"
    repo.mkdir(parents=True)
    mount.mkdir()
    config = compose_scene_config(repo)
    links = [
        "data/synthetic_data_generation",
        "data/tennis_multiview/raw/meiji_3cam",
        "data/tennis_multiview/processed/meiji_3cam",
    ]
    mirror = mount / repo.relative_to("/")
    for relative in links:
        source, target = repo / relative, mirror / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            source.rename(target)
        else:
            target.mkdir()
        source.parent.mkdir(parents=True, exist_ok=True)
        source.symlink_to(target, target_is_directory=True)
    raw = mirror / links[1] / "video_000"
    raw.mkdir()
    for name in ("cam0.mp4", "cam1.mp4"):
        (raw / name).write_bytes(name.encode())
    (mirror / links[2] / "projects.json").write_text("{}")
    policy = repo / ".agents/ops/storage_tier/config.toml"
    policy.parent.mkdir(parents=True)

    def write_policy(names: list[str]) -> None:
        policy.write_text(
            f"mirror_mount = {json.dumps(str(mount))}\n"
            "dataset_containers = []\ndataset_units = []\n"
            f"unmanaged_data_links = {json.dumps(names)}\n"
        )

    write_policy(links)
    monkeypatch.setattr(local_storage, "POLICY_CONFIG", policy, raising=False)
    original = os.path.ismount
    monkeypatch.setattr(
        os.path, "ismount", lambda path: Path(path) == mount or original(path)
    )
    roots = RuntimePathRoots.from_mapping(dict(config.roots), repository_root=repo)
    return repo, mount, mirror, links, config, PathResolver(roots), write_policy


@pytest.mark.parametrize("consumer", ["clip", "synthetic"])
@pytest.mark.parametrize("state", ["allowed", "undeclared", "mismatch", "unmounted"])
def test_consumers_keep_data_relative_inputs_through_declared_e_links(
    tiered, monkeypatch: pytest.MonkeyPatch, consumer: str, state: str
) -> None:
    repo, mount, mirror, links, config, resolver, write_policy = tiered
    relative = links[1] if consumer == "clip" else links[0]
    if state == "undeclared":
        write_policy([path for path in links if path != relative])
    elif state == "mismatch":
        (repo / relative).unlink()
        wrong = mirror / "data/wrong"
        wrong.mkdir()
        (repo / relative).symlink_to(wrong, target_is_directory=True)
    elif state == "unmounted":
        monkeypatch.setattr(os.path, "ismount", lambda path: False)

    def read_consumer() -> None:
        if consumer == "clip":
            layout = discover_clip_studio_layout(
                resolver, "tennis_multiview/raw/meiji_3cam/video_000"
            )
            assert layout.source_directory == mirror / relative / "video_000"
            assert [p.read_bytes() for p in layout.video_paths] == [
                b"cam0.mp4",
                b"cam1.mp4",
            ]
            assert layout.projects_path.read_text() == "{}"
            saved = ClipSource(layout.video_paths[0], "cam0").to_dict(resolver)
            assert saved["path"] == "tennis_multiview/raw/meiji_3cam/video_000/cam0.mp4"
            assert ClipSource.from_dict(saved, resolver).path == layout.video_paths[0]
        else:
            runtime = ScenePipelineConfiguration.from_config(config)
            assert runtime.request.source_video == mirror / relative / "raw/B00.mp4"
            assert (
                runtime.request.source_video.read_bytes()
                == b"configuration fixture; never decoded"
            )
            assert runtime.workspace.root == mirror / relative / "scenes/B00"
            assert runtime.workspace.root.resolve().is_relative_to(mirror / relative)

    if state == "allowed":
        read_consumer()
    else:
        with pytest.raises(PathContractError):
            read_consumer()


@pytest.mark.parametrize(
    "fault", ["nested-escape", "parent-escape", "inside-D", "other-role"]
)
def test_data_reference_exception_keeps_other_containment_guards(
    tiered, fault: str
) -> None:
    repo, _, mirror, links, _, resolver, _ = tiered
    relative = links[0]
    if fault == "nested-escape":
        outside = mirror / "outside.bin"
        outside.write_bytes(b"not an authorized input")
        (mirror / relative / "raw/escape.bin").symlink_to(outside)
        fragment = "synthetic_data_generation/raw/escape.bin"
    elif fault == "parent-escape":
        parent = resolver.resolve(PathRole.DATA, "synthetic_data_generation/raw")
    elif fault == "inside-D":
        alias = repo / relative
        alias.unlink()
        (repo / "data/wrong").mkdir()
        alias.symlink_to(repo / "data/wrong")
        fragment = "synthetic_data_generation"
    with pytest.raises(PathContractError):
        if fault == "parent-escape":
            resolver.resolve_beneath(PathRole.DATA, parent, "../scenes/B00")
        elif fault == "other-role":
            resolver.validate(PathRole.OUTPUT, mirror / relative / "raw/B00.mp4")
        else:
            resolver.resolve(PathRole.DATA, fragment)


def test_existing_physical_input_and_declared_file_alias_keep_logical_names(
    tiered,
) -> None:
    repo, _, mirror, links, _, resolver, write_policy = tiered
    physical = mirror / links[0] / "raw/B00.mp4"
    assert resolver.validate(PathRole.DATA, physical) == physical
    assert resolver.relative(PathRole.DATA, physical) == Path(
        "synthetic_data_generation/raw/B00.mp4"
    )
    relative = "data/tennis_multiview/processed/meiji_3cam.tar"
    (mirror / relative).write_bytes(b"known archive")
    (repo / relative).symlink_to(mirror / relative)
    write_policy([*links, relative])
    result = resolver.resolve(PathRole.DATA, relative.removeprefix("data/"))
    assert result == mirror / relative
    assert resolver.relative(PathRole.DATA, result) == Path(
        relative.removeprefix("data/")
    )
    assert result.read_bytes() == b"known archive"
