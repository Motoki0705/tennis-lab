"""Preparation validates source leakage and verifies immutable version receipts."""

import json
from pathlib import Path

import pytest

from src.tasks.plcs.data.preparation import prepare_motion_split, prepare_subset
from src.tasks.plcs.data.preparation_paths import preparation_resolver
from src.tasks.plcs.scripts.prepare_subset import PATH_BOUNDARY


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    rows = []
    assignments = {}
    for split in ("train", "val", "test"):
        names = []
        for index in range(4):
            name = f"{split}_{index}"
            names.append(name)
            rows.append(
                {"scene_id": name, "file": name, "motion_source": f"motion_{name}"}
            )
            assignments[f"motion_{name}"] = split
            directory = root / "scenes" / name
            directory.mkdir(parents=True)
            (directory / "payload").write_text(name)
        (root / f"{split}.txt").write_text("\n".join(names) + "\n")
    (root / "meta.json").write_text(json.dumps({"scenes": rows, "stats": {}}))
    (root / "scenes_meta.json").write_text(json.dumps(rows))
    (root / "config.yaml").write_text("test: true\n")
    (root / "split_info.json").write_text(
        json.dumps({"split_group": "motion_source", "source_assignments": assignments})
    )
    return root


def test_subset_reproducible_receipt_and_tamper(source: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    for destination in (first, second, first):
        prepare_subset(source, destination, seed=7, train=2, evaluation=2)
    assert (first / "dataset_version.json").read_bytes() == (
        second / "dataset_version.json"
    ).read_bytes()
    for split in ("train", "val", "test"):
        selected = (first / f"{split}.txt").read_text().splitlines()
        assert set(selected) <= set((source / f"{split}.txt").read_text().splitlines())
    (first / "train.txt").write_text("modified\n")
    with pytest.raises(ValueError, match="modified"):
        prepare_subset(source, first, seed=7, train=2, evaluation=2)


def test_parent_leakage_rejected_before_copy(source: Path, tmp_path: Path) -> None:
    rows = json.loads((source / "scenes_meta.json").read_text())
    rows[4]["motion_source"] = rows[0]["motion_source"]
    (source / "scenes_meta.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="leaks"):
        prepare_subset(source, tmp_path / "out", seed=7, train=1, evaluation=1)
    assert not (tmp_path / "out.building").exists()


@pytest.mark.parametrize(
    "seed,train,evaluation",
    [(-1, 2, 2), (True, 2, 2), (1, 0, 2), (1, 2, -1), (1, 99, 1)],
)
def test_invalid_subset_parameters(
    source: Path, tmp_path: Path, seed: int, train: int, evaluation: int
) -> None:
    with pytest.raises(ValueError):
        prepare_subset(
            source, tmp_path / "out", seed=seed, train=train, evaluation=evaluation
        )
    assert not (tmp_path / "out.building").exists()


def test_duplicate_motion_inventory_and_relative_paths(
    source: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="absolute"):
        prepare_motion_split(Path("relative"), tmp_path / "out", seed=1)
    rows = json.loads((source / "scenes_meta.json").read_text())
    rows.append(rows[0])
    (source / "scenes_meta.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="unique"):
        prepare_motion_split(source, tmp_path / "out", seed=1)
    assert not (tmp_path / "out.building").exists()


def test_motion_split_seed_and_receipt(source: Path, tmp_path: Path) -> None:
    for destination in (tmp_path / "first", tmp_path / "second", tmp_path / "first"):
        prepare_motion_split(source, destination, seed=42)
    assert (tmp_path / "first/dataset_version.json").read_bytes() == (
        tmp_path / "second/dataset_version.json"
    ).read_bytes()
    info = json.loads((tmp_path / "first/split_info.json").read_text())
    assert info["split_group"] == "motion_source"
    assert set(info["source_assignments"].values()) == {"train", "val", "test"}
    with pytest.raises(ValueError, match="recipe"):
        prepare_motion_split(source, tmp_path / "first", seed=43)


def test_paths_through_worktree_data_symlink(source: Path, tmp_path: Path) -> None:
    linked = tmp_path / "worktree-data"
    linked.symlink_to(source.parent, target_is_directory=True)
    input_path, output_path = linked / source.name, linked / "new-version"
    paths = PATH_BOUNDARY.validate(
        {"source": input_path, "destination": output_path},
        resolver=preparation_resolver(input_path, output_path),
    )
    assert paths.declared("source").path == source
    assert paths.declared("destination").path == tmp_path / "new-version"
