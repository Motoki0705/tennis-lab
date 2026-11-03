"""Small CPU contract tests for this experiment's build and verification gates."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import subprocess
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"scene_pair_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load("manifests")
transfer = load("transfer")


@pytest.fixture(scope="module")
def pool():
    header = {
        "schema": "court_dataset_subset_v1",
        "source_manifest": "pool.dataset.json",
        "source_manifest_sha256": "a" * 64,
        "selection_seed": 42,
    }
    rankings, lookup = {}, {}
    for scene, count in m.EVAL_COUNTS.items():
        rankings[scene] = {
            "train": [f"t-{scene}-{i}" for i in range(6000)],
            "val": [f"v-{scene}-{i}" for i in range(count)],
        }
        for split, ids in rankings[scene].items():
            for sample in ids:
                lookup[(scene, sample)] = {
                    "trajectory_id": f"trajectory-{split}",
                    "trajectory_group_id": f"group-{split}",
                }
    return header, rankings, lookup, {}


@pytest.fixture
def legacy(tmp_path, pool):
    header, rankings, _, _ = pool
    counts = m.read_json(m.ORIGINAL / "selection-report.json")["conditions"]
    for name, proof in counts.items():
        split = name.split(".")[1]
        m.write_json(
            tmp_path / name,
            {
                **header,
                "split": split,
                "samples": m.samples(rankings, split, proof["scene_counts"]),
            },
        )
    return tmp_path


def test_generated_manifest_boundaries_and_evaluation_population(pool):
    header, rankings, _, _ = pool
    documents = m.expected_manifests(header, rankings)
    prefixes = {
        "pair-b00-b02",
        "pair-b00-b03",
        "pair-b01-b02",
        "pair-b01-b03",
        "pair-b02-b03",
    }
    assert set(documents) == {
        f"{pair}.{split}.subset.json" for pair in prefixes for split in ("train", "val")
    } | {"eval-4scene.val.subset.json"}
    rows = documents["pair-b01-b03.train.subset.json"]["samples"]
    assert len(rows) == 5500
    assert [rows[i] for i in (0, 2749, 2750, 5499)] == [
        {"scene_id": "B01", "sample_id": "t-B01-0"},
        {"scene_id": "B01", "sample_id": "t-B01-2749"},
        {"scene_id": "B03", "sample_id": "t-B03-0"},
        {"scene_id": "B03", "sample_id": "t-B03-2749"},
    ]
    evaluation = documents["eval-4scene.val.subset.json"]["samples"]
    assert len(evaluation) == 2439
    assert [evaluation[i] for i in (0, 1055, 1056, 1699, 2161, 2438)] == [
        {"scene_id": "B00", "sample_id": "v-B00-0"},
        {"scene_id": "B00", "sample_id": "v-B00-1055"},
        {"scene_id": "B01", "sample_id": "v-B01-0"},
        {"scene_id": "B02", "sample_id": "v-B02-0"},
        {"scene_id": "B03", "sample_id": "v-B03-0"},
        {"scene_id": "B03", "sample_id": "v-B03-277"},
    ]


@pytest.mark.parametrize("split", ["train", "val"])
def test_p01_mismatch_blocks_new_outputs(legacy, pool, monkeypatch, split):
    path = legacy / f"scene-2.{split}.subset.json"
    document = m.read_json(path)
    document["samples"][0], document["samples"][1] = (
        document["samples"][1],
        document["samples"][0],
    )
    path.unlink()
    m.write_json(path, document)
    monkeypatch.setattr(m, "metadata", lambda _: pool)
    output = legacy / "new-manifests"
    args = argparse.Namespace(
        stage="create", metadata_root=legacy, manifest_root=output
    )
    report = {}
    with pytest.raises(ValueError, match=f"P01 {split} differs"):
        m.execute(args, report)
    assert report["p01"][split]["ordered_samples_equal"] is False
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["prefix", "duplicate"])
def test_verifier_rejects_membership_drift(legacy, pool, mutation):
    header, rankings, lookup, _ = pool
    expected = m.expected_manifests(header, rankings)
    actual = copy.deepcopy(expected)
    rows = actual["pair-b00-b02.train.subset.json"]["samples"]
    if mutation == "prefix":
        rows[0]["sample_id"] = "t-B00-2750"
        message = "prefix/count mismatch"
    else:
        rows[1] = rows[0].copy()
        message = "Duplicate subset"
    with pytest.raises(ValueError, match=message):
        m.verify(legacy, actual, expected, header, lookup)


@pytest.mark.parametrize("field", ["trajectory_id", "trajectory_group_id"])
def test_disjoint_frames_still_detect_trajectory_leakage(field):
    train = [{"scene_id": "B00", "sample_id": "first-frame"}]
    evaluation = [{"scene_id": "B00", "sample_id": "another-frame"}]
    first = {"trajectory_id": "first-trajectory", "trajectory_group_id": "first-group"}
    second = {
        "trajectory_id": "second-trajectory",
        "trajectory_group_id": "second-group",
        field: first[field],
    }
    result = m.overlaps(
        train,
        evaluation,
        {("B00", "first-frame"): first, ("B00", "another-frame"): second},
    )
    assert result["sample_overlap"] == 0
    assert result[f"{field}_overlap"] == 1
    with pytest.raises(ValueError, match="trajectory leakage"):
        m.require_disjoint(result)


def test_trajectory_identity_is_scoped_to_the_scene():
    train = [{"scene_id": "B00", "sample_id": "frame"}]
    evaluation = [{"scene_id": "B01", "sample_id": "frame"}]
    group = {"trajectory_id": "trajectory", "trajectory_group_id": "group"}
    result = m.overlaps(
        train, evaluation, {("B00", "frame"): group, ("B01", "frame"): group}
    )
    assert result == {
        "sample_overlap": 0,
        "trajectory_id_overlap": 0,
        "trajectory_group_id_overlap": 0,
    }
    m.require_disjoint(result)


def test_existing_remote_filename_is_refused():
    name = "pair-b00-b02.train.subset.json"
    with pytest.raises(ValueError, match="existing filename"):
        transfer.check_new_names([name], {name: {"id": "existing-drive-file"}})


def test_existing_output_directory_is_refused(legacy, pool, monkeypatch):
    monkeypatch.setattr(m, "metadata", lambda _: pool)
    output = legacy / "outputs"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("prior content")
    args = argparse.Namespace(
        stage="create", metadata_root=legacy, manifest_root=output
    )
    with pytest.raises(ValueError, match="existing manifest output directory"):
        m.execute(args, {})
    assert sentinel.read_text() == "prior content"


@pytest.mark.parametrize("mutation", ["failed-verification", "changed-file"])
def test_publish_checks_verified_bytes_before_accessing_drive(
    tmp_path, monkeypatch, mutation
):
    name = "pair-b00-b02.train.subset.json"
    (tmp_path / name).write_text("current manifest")
    audit = tmp_path / "audit.json"
    m.write_json(
        audit,
        {
            "status": "failed" if mutation == "failed-verification" else "passed",
            "stage": "verify",
            "manifests": {name: {"sha256": "a" * 64}},
        },
    )
    calls = []
    monkeypatch.setattr(transfer, "inventory", lambda *args: calls.append(args))
    args = argparse.Namespace(audit=audit, local_root=tmp_path)
    message = (
        "successful independent verify"
        if mutation == "failed-verification"
        else "changed after verification"
    )
    with pytest.raises(ValueError, match=message):
        transfer.publish(args)
    assert calls == []


def test_metadata_resume_rejects_corrupt_local_bytes(tmp_path):
    name = "index.npz"
    (tmp_path / name).write_bytes(b"broken")
    observed = {name: {"type": "file", "size_bytes": 6, "hashes": {"md5": "0" * 32}}}
    with pytest.raises(ValueError, match="Content mismatch"):
        transfer.missing_files(tmp_path, [name], observed)


def test_metadata_download_reports_every_failed_batch(tmp_path, monkeypatch):
    class FailedBackend:
        def __init__(self, **kwargs):
            pass

        def run(self, arguments, **kwargs):
            raise ValueError(arguments[-1])

    monkeypatch.setattr(transfer, "RcloneBackend", FailedBackend)
    paths = [f"store-{i}/index.npz" for i in range(4)]
    observed = {name: {"id": f"drive-id-{i}"} for i, name in enumerate(paths)}
    with pytest.raises(ExceptionGroup) as caught:
        transfer.download_ids(tmp_path / "config", tmp_path, paths, observed)
    assert {str(error) for error in caught.value.exceptions} == {
        str((tmp_path / name).resolve()) for name in paths
    }


def test_inventory_surfaces_the_cli_failure_reason(tmp_path, monkeypatch):
    def failed(argv, **kwargs):
        output = '{"error":"Ambiguous Drive path: inspect the listed IDs"}'
        if kwargs.get("check"):
            raise subprocess.CalledProcessError(1, argv, output=output)
        return subprocess.CompletedProcess(argv, 1, stdout=output, stderr="")

    monkeypatch.setattr(transfer.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="Ambiguous Drive path"):
        transfer.inventory(tmp_path / "config", tmp_path / "inventory.json")
