from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from src.utils.inference_benchmark_inputs import PinnedBenchmarkInputs


def _write_json(path: Path, value: object) -> str:
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return sha256(data).hexdigest()


def _published(directory: Path, *, swap: bool = False) -> PinnedBenchmarkInputs:
    rows = []
    for key in "abcd":
        path = directory / f"{key}.bin"
        path.write_bytes(key.encode())
        rows.append(
            {"id": key, "path": path.name, "sha256": sha256(key.encode()).hexdigest()}
        )
    if swap:
        rows[0], rows[1] = rows[1], rows[0]
    index = _write_json(directory / "frames.json", rows)
    manifest = _write_json(
        directory / "manifest.json",
        {
            "indexes": {
                "frames": {"path": "frames.json", "sha256": index, "records": 4}
            },
        },
    )
    timing_path = directory / "timing.json"
    protocol = _write_json(
        timing_path,
        {
            "schema_version": 1,
            "input_manifest_sha256": manifest,
            "warmup": 1,
            "iterations_per_block": 3,
            "blocks": 2,
            "schedules": {
                "frames": {
                    "records": 4,
                    "batch_sizes": {
                        "1": sha256(
                            b'[["a"],["b"],["c"],["d"],["a"],["b"]]'
                        ).hexdigest(),
                        "2": sha256(
                            b'[["a","b"],["c","d"],["a","b"],["c","d"],["a","b"],["c","d"]]'
                        ).hexdigest(),
                    },
                }
            },
        },
    )
    return PinnedBenchmarkInputs(
        directory, timing_path, manifest_sha256=manifest, timing_sha256=protocol
    )


def test_pinned_schedule_rejects_reordered_loaded_batches(tmp_path: Path) -> None:
    fixed = _published(tmp_path)
    schedule = fixed.schedule("frames", 2)
    assert schedule.batch_ids == (("a", "b"), ("c", "d"))
    assert schedule.protocol.iterations_per_block == 3 and schedule.protocol.blocks == 2
    schedule.require_batches(
        [tuple(row["id"] for row in fixed.rows("frames")[:2]), ("c", "d")]
    )
    with pytest.raises(ValueError, match="input IDs or order"):
        schedule.require_batches((("b", "a"), ("c", "d")))
    assert fixed.verified_asset(fixed.rows("frames")[0]).read_bytes() == b"a"
    (tmp_path / "a.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset SHA-256 mismatch"):
        fixed.verified_asset(fixed.rows("frames")[0])


@pytest.mark.parametrize("path", ["manifest.json", "timing.json"])
def test_reads_require_both_published_hashes(tmp_path: Path, path: str) -> None:
    fixed = _published(tmp_path)
    changed = tmp_path / path
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch at .*" + path):
        PinnedBenchmarkInputs(
            tmp_path,
            tmp_path / "timing.json",
            manifest_sha256=fixed.manifest_sha256,
            timing_sha256=fixed.timing_sha256,
        )


def test_schedule_rejects_manifest_order_that_disagrees_with_timing_ids(
    tmp_path: Path,
) -> None:
    fixed = _published(tmp_path, swap=True)
    with pytest.raises(ValueError, match="pinned frames B=1 timing order"):
        fixed.schedule("frames", 1)
