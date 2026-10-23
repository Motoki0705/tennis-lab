"""Check the interrupted, resumed and independently fetched small L4 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.synthetic_data_generation.dataset.court.sample_store import open_court_store
from src.synthetic_data_generation.dataset.court.scale.publication import (
    load_dataset_version,
    read_json,
)
from src.utils.data.image_record_store import file_sha256


def inventory(root: Path) -> dict[str, str]:
    owners = {
        "requests",
        "plans",
        "curves",
        "assets",
        "stores",
        "cache-metrics",
        "memberships",
        "reports",
        "completed-requests",
    }
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and (
            path.relative_to(root).parts[0] in owners
            or path.name.endswith(".dataset.json")
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interrupted-root", type=Path, required=True)
    parser.add_argument("--resumed-root", type=Path, required=True)
    parser.add_argument("--fetched-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    request_sha = read_json(args.evidence_root / "request.json")["request_sha256"]
    isolation = read_json(args.evidence_root / "isolation.json")
    assert (
        isolation["original_root_absent"]
        and not Path(isolation["original_root"]).exists()
    )
    assert (
        isolation["resume_root_absent_before"] and isolation["fetch_root_absent_before"]
    )
    assert isolation["completed_drive_receipt_absent"]

    initial_attempts = list((args.interrupted_root / "attempts").glob("*.json"))
    resumed_attempts = list((args.resumed_root / "attempts").glob("*.json"))
    assert len(initial_attempts) == len(resumed_attempts) == 1
    initial = read_json(initial_attempts[0])
    resumed = read_json(resumed_attempts[0])
    assert initial["outcome"] == "interrupted" and resumed["outcome"] == "completed"
    assert initial["request_sha256"] == resumed["request_sha256"] == request_sha
    assert len(initial["caches"]) == 1
    first_key = initial["caches"][0]["cache_key"]
    assert first_key == isolation["acknowledged_cache_keys"][0]
    assert initial["caches"][0] == {
        "cache_key": first_key,
        "reused": False,
        "restored_from_drive": False,
        "drive_acknowledged": True,
    }
    assert not list(args.interrupted_root.glob("*.dataset.json"))
    assert not list((args.interrupted_root / "completed-requests").glob("*.json"))
    events = resumed["caches"]
    assert len(events) > 1 and len({event["cache_key"] for event in events}) == len(
        events
    )
    restored = [event for event in events if event["cache_key"] == first_key]
    assert restored == [
        {
            "cache_key": first_key,
            "reused": True,
            "restored_from_drive": True,
            "drive_acknowledged": True,
        }
    ]
    for event in events:
        if event["cache_key"] != first_key:
            assert not event["reused"] and not event["restored_from_drive"]
        assert event["drive_acknowledged"]
    for relative in (
        f"stores/{first_key}/completion.json",
        f"cache-metrics/{first_key}.json",
    ):
        assert file_sha256(args.interrupted_root / relative) == file_sha256(
            args.resumed_root / relative
        )

    versions = list(args.fetched_root.glob("*.dataset.json"))
    assert len(versions) == 1
    version_path = versions[0]
    version = load_dataset_version(version_path, verify_images=True)
    assert version == load_dataset_version(
        args.resumed_root / version_path.name, verify_images=True
    )
    assert version["schema"] == "court_scale_dataset_version_v2"
    assert version["split"] == "train" and version["generation_seed"] == 42
    assert version["method"] == "stratified_counter"
    assert version["request_sha256"] == request_sha
    assert [scene["scene_id"] for scene in version["scenes"]] == [
        "B00",
        "B01",
        "B02",
        "B03",
    ]
    receipt = read_json(args.fetched_root / f"completed-requests/{request_sha}.json")
    assert receipt == {"dataset_version_path": version_path.name}
    request = read_json(args.fetched_root / version["request_path"])
    assert all(
        scene["trajectory_count"] == 6 and scene["density_level"] == 0
        for scene in request["scenes"]
    )

    original_files = inventory(args.resumed_root)
    fetched_files = inventory(args.fetched_root)
    assert original_files == fetched_files
    inventory_sha = hashlib.sha256(
        json.dumps(fetched_files, sort_keys=True).encode()
    ).hexdigest()
    report_path = args.fetched_root / f"reports/{request_sha}.json"
    report = read_json(report_path)
    assert report["dataset_version_sha256"] == file_sha256(version_path)
    assert report["renderer_release"] == resumed["renderer_release"]
    for attempt in (initial, resumed):
        release = attempt["renderer_release"]
        assert (
            release["renderer_context_released"]
            and release["compute_processes_after"] == []
        )
        assert release["before"]["name"] == release["after"]["name"] == "NVIDIA L4"

    store_count = decoded = cache_count = 0
    for scene in version["scenes"]:
        rows = 0
        assert scene["counts"]["pending"] == scene["counts"]["failed"] == 0
        cache_count += len(scene["render_caches"])
        for entry in scene["stores"]:
            store = open_court_store(args.fetched_root / entry["path"])
            store.validate(decode=True)
            rows += len(store)
            store_count += 1
        assert rows == scene["counts"]["qa_accepted"]
        decoded += rows
    assert cache_count == len(events)
    result = {
        "schema": "court_scale_smoke_verification_v1",
        "request_sha256": request_sha,
        "dataset_version_path": version_path.name,
        "dataset_version_sha256": file_sha256(version_path),
        "generation_report_path": report_path.relative_to(args.fetched_root).as_posix(),
        "checks": {
            "original_output_root_unavailable": True,
            "fresh_resume_and_fetch_roots": True,
            "interruption_has_no_completed_receipt": True,
            "completed_cache_restored_without_rendering": True,
            "only_unacknowledged_caches_generated": True,
            "drive_roundtrip_inventory_identical": True,
            "load_dataset_version_verify_images": True,
            "every_sample_store_validate_decode": True,
            "renderer_contexts_released": True,
        },
        "interrupted_cache_key": first_key,
        "cache_count": cache_count,
        "restored_cache_count": len(restored),
        "new_cache_count": len(events) - len(restored),
        "sample_store_count": store_count,
        "decoded_image_count": decoded,
        "compared_file_count": len(fetched_files),
        "compared_inventory_sha256": inventory_sha,
        "torch_allocator_peak_measured": False,
    }
    temporary = args.evidence_root.parent / ".court-scale-verification.tmp"
    with temporary.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.evidence_root / "verification.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
