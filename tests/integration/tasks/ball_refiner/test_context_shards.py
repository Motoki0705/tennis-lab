"""Whole-clip jobs assemble into the existing reader, with no partial recovery."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import src.tasks.ball_refiner.data.context_shards as shard_module
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.context_cache import ContextCache
from src.tasks.ball_refiner.data.context_shards import (
    generate_context_shard,
    merge_context_shards,
    read_context_plan,
    write_context_plan,
)
from src.tasks.ball_refiner.data.evidence_cache import (
    EvidenceCache,
    generate_evidence_cache,
)
from tests.integration.tasks.ball_refiner.test_context_cache import (
    FakeProducer as BaseProducer,
)
from tests.integration.tasks.ball_refiner.test_evidence_cache import (
    cache_inputs as cache_inputs,
)
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


class FakeProducer(BaseProducer):
    def identity(self):
        return {**super().identity(), "shape": (17, 3)}


@pytest.fixture
def shard_setup(cache_inputs, tmp_path):
    write_store_clip(cache_inputs["store_directory"], "tracknet/game2/clip2", [frame(i, ball()) for i in range(6)])
    evidence = EvidenceCache(generate_evidence_cache(**cache_inputs), BallFrameStore(cache_inputs["store_directory"]))
    producer = FakeProducer()
    plan_path = write_context_plan(evidence, producer=producer, output=tmp_path / "plan")
    shards = tuple(generate_context_shard(
        evidence, plan_path=plan_path, shard_index=index, producer=producer, output=tmp_path / f"shard-{index}",
    ) for index in range(len(evidence.clip_ids)))
    return evidence, plan_path, shards


def test_whole_clip_plan_and_order_independent_merge_preserve_every_npz(shard_setup, tmp_path):
    evidence, plan_path, shards = shard_setup
    plan = read_context_plan(plan_path, evidence)
    assert [row["frames"] for row in plan["shards"]] == [9, 6]
    assert [row["clip_id"] for row in plan["shards"]] == list(evidence.clip_ids)
    originals = [(directory / "manifest.json").read_bytes() for directory in shards]
    directory = merge_context_shards(evidence, plan_path=plan_path, shards=shards[::-1], output=tmp_path / "assembled")
    merged = ContextCache(directory, evidence)
    merged.require_clips(evidence.clip_ids)
    assert merged.manifest["selection"]["scope"] == "all_evidence"
    assert [row["clip_id"] for row in merged.manifest["assembly"]["shards"]] == list(evidence.clip_ids)
    for clip_id, source in zip(evidence.clip_ids, shards, strict=True):
        expected = ContextCache(source, evidence).load(clip_id)
        actual = merged.load(clip_id)
        for name, array in expected.arrays.arrays().items():
            np.testing.assert_array_equal(array, actual.arrays.arrays()[name])
        record = next(item for item in merged.manifest["clips"] if item["clip"]["clip_id"] == clip_id)
        assert (directory / record["file"]).read_bytes() == (source / record["file"]).read_bytes()
        assert (directory / record["file"]).stat().st_ino != (source / record["file"]).stat().st_ino
    assert originals == [(path / "manifest.json").read_bytes() for path in shards]
    with pytest.raises(FileExistsError):
        merge_context_shards(evidence, plan_path=plan_path, shards=shards, output=directory)
    with pytest.raises(FileExistsError):
        write_context_plan(evidence, producer=FakeProducer(), output=plan_path.parent)


@pytest.mark.parametrize("change", ["missing", "repeated_path", "duplicate_clip", "incomplete", "model", "code", "rgb", "jpeg_identity", "receipt", "checksum"])
def test_assembly_rejects_missing_duplicate_incompatible_and_corrupt_shards(shard_setup, tmp_path, change):
    evidence, plan_path, shards = shard_setup
    if change == "missing":
        shards = shards[:-1]
    elif change == "repeated_path":
        shards = (shards[0], shards[0])
    elif change == "duplicate_clip":
        duplicate = generate_context_shard(
            evidence, plan_path=plan_path, shard_index=0, producer=FakeProducer(), output=tmp_path / "duplicate",
        )
        shards = (shards[0], duplicate)
    else:
        path = shards[-1] / "manifest.json"
        manifest = json.loads(path.read_text())
        if change == "incomplete":
            manifest["status"] = "building"
        elif change == "model":
            manifest["model_identity"]["revision"] = 99
        elif change == "code":
            manifest["generator_sha256"]["context_arrays.py"] = "bad"
        elif change == "rgb":
            manifest["rgb_condition"] = "occluded"
        elif change == "jpeg_identity":
            manifest["clips"][0]["jpeg_shard_sha256"] = "bad"
        elif change == "receipt":
            manifest["clips"][0]["execution"]["tracking_frames"] = 0
        else:
            manifest["clips"][0]["sha256"] = "bad"
        path.write_text(json.dumps(manifest))
    output = tmp_path / "assembled"
    with pytest.raises(ValueError):
        merge_context_shards(evidence, plan_path=plan_path, shards=shards, output=output)
    if output.exists():
        assert json.loads((output / "manifest.json").read_text())["status"] == "building"
        with pytest.raises(ValueError, match="Incomplete"):
            ContextCache(output, evidence)


@pytest.mark.parametrize("change", ["omit", "duplicate", "frames", "order", "code", "evidence", "model"])
def test_generation_rejects_changed_plan_or_models_before_producing(shard_setup, tmp_path, change):
    evidence, plan_path, _ = shard_setup
    plan = json.loads(plan_path.read_text())
    if change == "omit":
        plan["shards"].pop()
    elif change == "duplicate":
        plan["shards"][1] = plan["shards"][0]
    elif change == "frames":
        plan["shards"][0]["frames"] -= 1
    elif change == "order":
        plan["shards"].reverse()
    elif change == "code":
        plan["identity"]["generator_sha256"]["context_cache.py"] = "bad"
    elif change == "evidence":
        plan["identity"]["evidence"]["manifest_sha256"] = "bad"
    else:
        plan["identity"]["model_identity"]["revision"] = 99
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / "new-shard"
    with pytest.raises(ValueError):
        generate_context_shard(evidence, plan_path=plan_path, shard_index=0, producer=FakeProducer(), output=output)
    assert not output.exists()


@pytest.mark.parametrize("index", [-1, 2, True, .5])
def test_shard_index_cannot_select_outside_or_implicitly_convert(shard_setup, tmp_path, index):
    evidence, plan_path, _ = shard_setup
    with pytest.raises(ValueError, match="Shard index"):
        generate_context_shard(evidence, plan_path=plan_path, shard_index=index, producer=FakeProducer(), output=tmp_path / "bad")


@pytest.mark.parametrize("change", ["jpeg", "manifest", "plan", "source_npz", "destination_npz", "store"])
def test_mutation_during_copy_never_publishes_a_complete_cache(shard_setup, tmp_path, monkeypatch, change):
    evidence, plan_path, shards = shard_setup
    original = shard_module.shutil.copyfileobj
    calls = 0

    def mutate(src, dst):
        nonlocal calls
        original(src, dst)
        calls += 1
        if calls != len(shards):
            return
        first_record = json.loads((shards[0] / "manifest.json").read_text())["clips"][0]
        paths = {
            "jpeg": evidence.store.directory / "shards/clip-00000.bin",
            "manifest": shards[0] / "manifest.json", "plan": plan_path,
            "source_npz": Path(src.name),
            "destination_npz": tmp_path / "assembled" / first_record["file"],
            "store": evidence.store.directory / "metadata.json",
        }
        with paths[change].open("ab") as stream:
            stream.write(b" ")

    monkeypatch.setattr(shard_module.shutil, "copyfileobj", mutate)
    output = tmp_path / "assembled"
    with pytest.raises(ValueError):
        merge_context_shards(evidence, plan_path=plan_path, shards=shards, output=output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "building"


def test_merge_cli_reads_in_a_separate_cpu_process(shard_setup, tmp_path):
    evidence, plan_path, shards = shard_setup
    output = tmp_path / "from-cli"
    command = [sys.executable, "-m", "src.tasks.ball_refiner.scripts.context_shards", "merge",
               "--store", str(evidence.store.directory), "--evidence", str(evidence.directory),
               "--plan", str(plan_path), "--output", str(output)]
    for path in reversed(shards):
        command.extend(("--shard", str(path)))
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert ContextCache(output, evidence).clip_ids == evidence.clip_ids
    repeated = subprocess.run(command, text=True, capture_output=True, check=False)
    assert repeated.returncode != 0 and "Output must be new" in repeated.stderr


def test_shard_cli_rejects_relative_paths_before_loading_models(tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "src.tasks.ball_refiner.scripts.context_shards", "plan",
        "--store", "relative", "--evidence", str(tmp_path), "--output", str(tmp_path / "out"),
        "--scene-config", str(tmp_path / "scene.yaml"), "--max-tracks", "1024",
    ], capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "All paths must be absolute" in result.stderr
    assert not (tmp_path / "out").exists()
