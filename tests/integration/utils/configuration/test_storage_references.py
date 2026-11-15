"""D: logical input contracts across declared E: aliases, via real consumers."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.scripts.edit_alignment import PATH_BOUNDARY
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
    subprocess.run(["git", "init", "--quiet", "--template=", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        check=True,
    )
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
    monkeypatch.setattr(local_storage, "PROJECT_ROOT", repo)
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


@pytest.mark.parametrize("fault", ["nested-escape", "parent-escape", "other-role"])
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


@pytest.mark.parametrize("consumer", ["alignment", "clip"])
@pytest.mark.parametrize(
    "state", ["allowed", "undeclared", "mismatch", "unmounted", "nested-escape"]
)
def test_worktree_absolute_inputs_keep_the_standard_data_contract(
    tiered, monkeypatch: pytest.MonkeyPatch, consumer: str, state: str
) -> None:
    repo, mount, mirror, links, _, resolver, write_policy = tiered
    worktree = repo / ".claude/worktrees/editor"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "worktree",
            "add",
            "--quiet",
            "--detach",
            "--no-checkout",
            str(worktree),
        ],
        check=True,
    )
    (worktree / "data").symlink_to(repo / "data", target_is_directory=True)
    relative = links[0] if consumer == "alignment" else links[1]
    if state == "undeclared":
        write_policy([name for name in links if name != relative])
    elif state == "mismatch":
        (repo / relative).unlink()
        wrong = mirror / "data/wrong"
        wrong.mkdir()
        (repo / relative).symlink_to(wrong, target_is_directory=True)
    elif state == "unmounted":
        monkeypatch.setattr(os.path, "ismount", lambda path: False)
    policy = worktree / ".agents/ops/storage_tier/config.toml"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(local_storage.POLICY_CONFIG.read_bytes())
    monkeypatch.setattr(local_storage, "PROJECT_ROOT", worktree)
    monkeypatch.setattr(local_storage, "POLICY_CONFIG", policy)
    suffix = "scenes/B00" if consumer == "alignment" else "video_000/cam0.mp4"
    target = mirror / relative / suffix
    if consumer == "alignment":
        target.mkdir(parents=True, exist_ok=True)
        (target / "proof.txt").write_bytes(b"alignment input")
    if state == "nested-escape":
        outside = mount / "outside"
        if target.is_dir():
            outside.mkdir()
            (outside / "proof.txt").write_bytes(b"alignment input")
            target.rename(target.with_name("saved"))
        else:
            outside.write_bytes(b"cam0.mp4")
            target.unlink()
        target.symlink_to(outside, target_is_directory=consumer == "alignment")
    supplied = worktree / relative / suffix

    def read_consumer() -> None:
        if consumer == "alignment":
            paths = PATH_BOUNDARY.validate(
                {"data_root": resolver.roots.data_root, "scene_root": supplied},
                resolver=resolver,
            )
            scene = paths.declared("scene_root").path
            assert scene == target
            assert (scene / "proof.txt").read_bytes() == b"alignment input"
        else:
            saved = ClipSource(supplied, "cam0").to_dict(resolver)
            assert saved["path"] == "tennis_multiview/raw/meiji_3cam/video_000/cam0.mp4"
            restored = ClipSource.from_dict(saved, resolver)
            assert restored.path == target
            assert restored.path.read_bytes() == b"cam0.mp4"
        assert (
            resolver.relative(PathRole.DATA, supplied)
            == Path(relative.removeprefix("data/")) / suffix
        )

    if state == "allowed":
        read_consumer()
    else:
        with pytest.raises(PathContractError):
            read_consumer()


@pytest.mark.parametrize("target_kind", ["internal", "external", "E-shaped"])
@pytest.mark.parametrize("git_state", ["available", "missing", "metadata-missing"])
def test_custom_data_root_keeps_original_containment(
    tiered, monkeypatch: pytest.MonkeyPatch, target_kind: str, git_state: str
) -> None:
    repo, mount, _, _, _, resolver, _ = tiered
    custom = repo.parent / "custom/data"
    custom.mkdir(parents=True)
    if target_kind == "internal":
        target = custom / "local"
        monkeypatch.setattr(os.path, "ismount", lambda path: False)
    elif target_kind == "external":
        target = repo.parent / "outside"
    else:
        target = (
            mount / custom.parent.relative_to("/") / "data/synthetic_data_generation"
        )
    target.mkdir(parents=True)
    (target / "input.txt").write_bytes(b"custom input")
    alias = custom / "synthetic_data_generation"
    alias.symlink_to(target, target_is_directory=True)
    custom_resolver = PathResolver(replace(resolver.roots, data_root=custom))
    if git_state == "missing":
        empty_path = repo.parent / "empty-PATH"
        empty_path.mkdir()
        monkeypatch.setenv("PATH", str(empty_path))
    elif git_state == "metadata-missing":
        (repo / ".git").rename(repo / ".git-retained")
    if target_kind == "internal":
        result = custom_resolver.resolve(
            PathRole.DATA, "synthetic_data_generation/input.txt"
        )
        assert result.read_bytes() == b"custom input"
        assert custom_resolver.relative(PathRole.DATA, alias / "input.txt") == Path(
            "local/input.txt"
        )
    else:
        with pytest.raises(PathContractError):
            custom_resolver.resolve(
                PathRole.DATA, "synthetic_data_generation/input.txt"
            )


@pytest.mark.parametrize(
    "spelling", ["arbitrary-prefix", "chained-data", "linked-parent"]
)
def test_only_direct_standard_worktree_root_aliases_are_authorized(
    tiered, spelling: str
) -> None:
    repo, _, _, _, _, resolver, _ = tiered
    worktree = repo / ".claude/worktrees/editor"
    worktree.mkdir(parents=True)
    if spelling == "arbitrary-prefix":
        prefix = repo / "other"
        prefix.mkdir()
    elif spelling == "linked-parent":
        actual = repo / "actual"
        actual.mkdir()
        worktree.rmdir()
        worktree.symlink_to(actual, target_is_directory=True)
        prefix = worktree
    else:
        intermediate = repo / "data-chain"
        intermediate.symlink_to(repo / "data", target_is_directory=True)
        prefix = worktree
    (prefix / "data").symlink_to(
        intermediate if spelling == "chained-data" else repo / "data",
        target_is_directory=True,
    )
    with pytest.raises(PathContractError):
        resolver.validate(
            PathRole.DATA, prefix / "data/synthetic_data_generation/raw/B00.mp4"
        )


def test_runtime_worktree_uses_main_data_even_before_its_shared_link_is_created(
    tiered, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _, mirror, _, _, resolver, _ = tiered
    runtime = repo / ".claude/worktrees/runtime"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "worktree",
            "add",
            "--quiet",
            "--detach",
            "--no-checkout",
            str(runtime),
        ],
        check=True,
    )
    monkeypatch.setattr(local_storage, "PROJECT_ROOT", runtime)
    relative = "synthetic_data_generation/raw/B00.mp4"
    result = resolver.resolve(PathRole.DATA, relative)
    assert result == mirror / "data" / relative
    assert result.read_bytes() == b"configuration fixture; never decoded"


def test_contained_declared_link_keeps_ordinary_path_rules_without_git(
    tiered, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _, _, links, _, resolver, _ = tiered
    target = repo / "data/local"
    target.mkdir()
    (target / "input.bin").write_bytes(b"ordinary contained input")
    alias = repo / links[0]
    alias.unlink()
    alias.symlink_to(target, target_is_directory=True)
    empty_path = repo.parent / "empty-PATH"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    result = resolver.resolve(PathRole.DATA, "synthetic_data_generation/input.bin")
    assert result.read_bytes() == b"ordinary contained input"
    assert resolver.relative(PathRole.DATA, alias / "input.bin") == Path(
        "local/input.bin"
    )


def test_repository_identity_is_cached_but_policy_and_alias_are_revalidated(
    tiered, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _, mirror, links, _, resolver, write_policy = tiered
    query = local_storage.subprocess.check_output
    calls = []

    def observed(*args, **kwargs):
        calls.append(args)
        return query(*args, **kwargs)

    monkeypatch.setattr(local_storage.subprocess, "check_output", observed)
    relative = "synthetic_data_generation/raw/B00.mp4"
    for _ in range(2):
        resolved = resolver.resolve(PathRole.DATA, relative)
        assert resolved.read_bytes() == b"configuration fixture; never decoded"
        assert resolver.relative(PathRole.DATA, resolved) == Path(relative)
    assert len(calls) == 1
    write_policy([name for name in links if name != links[0]])
    with pytest.raises(PathContractError):
        resolver.resolve(PathRole.DATA, relative)
    write_policy(links)
    alias, expected = repo / links[0], mirror / links[0]
    alias.unlink()
    alias.symlink_to(os.path.relpath(expected, alias.parent), target_is_directory=True)
    with pytest.raises(PathContractError):
        resolver.resolve(PathRole.DATA, relative)
    assert len(calls) == 1
