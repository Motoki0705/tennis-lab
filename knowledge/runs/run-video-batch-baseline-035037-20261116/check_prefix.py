"""CPU-only check of benchmark cache preparation with real store/codec boundaries."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))
sys.dont_write_bytecode = True

from src.tennis_scene.generate_dataset.manifest import load_dataset_manifest
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


@dataclass(frozen=True)
class Output:
    values: NDArray[np.float32]


def check(root: Path) -> None:
    specification = importlib.util.spec_from_file_location("c1_benchmark", Path(__file__).with_name("benchmark.py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    dataset = root / "verification-prefix"
    dataset.mkdir(exist_ok=False)
    shutil.copyfile(root / "input/dataset.json", dataset / "dataset.json")
    expected = {}
    codec = ArtifactCodec(Output)
    for record in load_dataset_manifest(dataset).clips.values():
        directory = dataset / record.path
        directory.mkdir(parents=True)
        shutil.copyfile(root / "input" / record.path / "clip.json", directory / "clip.json")
        store = ClipStore(directory / "annotations/tennis_scene", {"fake_cpu_clip": record.clip_id})
        courts = {}
        for camera in range(4):
            name = f"court_detection/cam{camera}"
            courts[name] = store.publish(name, Output(np.array([camera, camera + 7], np.float32)), codec,
                schema="fake", version=1, identity={"node": name}, dependencies={}, provenance={"origin": "fake_cpu"})
        calibration = store.publish("court_calibration", Output(np.array([11, -3], np.float32)), codec,
            schema="fake", version=1, identity={"node": "court_calibration"}, dependencies=courts, provenance={"origin": "fake_cpu"})
        ball = store.publish("ball_detection/cam0", Output(np.array([5, 9], np.float32)), codec,
            schema="fake", version=1, identity={"node": "ball"}, dependencies={}, provenance={"origin": "fake_cpu"})
        np.savez(store.root / "scene.npz", values=np.array([13, -8]))
        (store.root / "scene.metadata.json").write_text("{}")
        store.record_export("scene", {"scene": store.root / "scene.npz", "metadata": store.root / "scene.metadata.json"},
                            {"court_calibration": calibration, "ball_detection/cam0": ball})
        expected[record.clip_id] = {**courts, "court_calibration": calibration}
    backup = root / "verification-prefix-backup"
    module.prepare_prefix(dataset, backup)
    for record in load_dataset_manifest(dataset).clips.values():
        store = ClipStore(dataset / record.path / "annotations/tennis_scene", {"fake_cpu_clip": record.clip_id})
        assert store.references() == expected[record.clip_id]
        assert json.loads(store.index_path.read_text())["exports"] == {}
        assert store.active("ball_detection/cam0") is None
        assert not (store.root / "annotation.json").exists()
        for reference in expected[record.clip_id].values():
            store.load(reference, codec)
            relative = Path(reference.path).parent
            for path in (store.root / relative).iterdir():
                assert path.read_bytes() == (backup / record.clip_name / relative / path.name).read_bytes()
    (root / "prefix-preparation-validation.json").write_text(json.dumps({
        "scope": "CPU fake artifacts only; not a real GPU prefix/cache-hit measurement", "clips": 2,
        "retained_per_clip": 5, "bytes_and_references_preserved": True, "downstream_and_exports_absent": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    if not root.is_relative_to((REPOSITORY / "outputs").resolve()):
        raise ValueError("Validation must stay below owned outputs")
    check(root)
