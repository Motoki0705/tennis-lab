"""Verify live dataset pointers and summarize the two measured publication passes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.imports.external_ball_geometry import (
    SAVED_GEOMETRY,
    SHARED_GEOMETRY,
)
from src.tennis_scene.pipeline.imports.external_ball_positions import (
    STATUSES,
    ExternalBallPositions,
)
from src.tennis_scene.pipeline.imports.external_ball_publication import (
    PublicationReceipt,
    ScenePointers,
)
from src.tennis_scene.pipeline.imports.external_ball_update import REJECTION_CODES
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_index import (
    annotation_scene_path,
    read_component_descriptor,
)

BALL_FIELDS = {"ball_uv", "ball_vis", "ball_3d", "ball_3d_valid", "ball_rejection_code"}


def verify_clip(dataset: Path, result: dict) -> tuple[dict, np.ndarray]:
    video, clip = result["clip_id"].split("/")
    root = dataset / "videos" / video / "clips" / clip / "annotations/tennis_scene"
    live = ScenePointers.read(root)
    index, marker = live.index(), live.marker()
    ref = result["scene_assembly"]
    assert marker is not None
    assert (
        live.revision() == ref["artifact_id"]
        and index["artifacts"]["scene_assembly"] == ref
    )
    receipt = PublicationReceipt.read(root, ref["artifact_id"])
    assert live == receipt.after
    assert receipt.pointer_digests() == result["pointer_digests"]
    archive = annotation_scene_path(root, marker)
    assert archive.parent.name == ref["artifact_id"]
    metadata = json.loads(archive.with_suffix(".metadata.json").read_text())
    ball_metadata = metadata["external_ball"]
    assert ball_metadata["geometry_provenance"]["origin"] == result["geometry"]
    before = receipt.before.index()
    original = before["artifacts"].get("scene_assembly")
    for node, reference in before["artifacts"].items():
        if node != "scene_assembly":
            assert index["artifacts"][node] == reference
    store = ClipStore(root, index["source"], memory_entries=0)
    positions = store.load(
        store.references()["external_ball_positions"],
        ArtifactCodec(ExternalBallPositions),
    )
    ball = store.load(
        store.references()["ball_triangulation"],
        ArtifactCodec(BallReconstructionResult),
    )
    assert positions.sha256 == result["input_sha256"]
    assert ball_metadata["status_codes"] == positions.status.tolist()
    assert ball_metadata["status_labels"] == list(STATUSES)
    np.testing.assert_array_equal(ball.visibility, positions.usable)
    np.testing.assert_array_equal(
        ball.trajectory.inliers, positions.usable & ball.trajectory.valid[None]
    )
    with np.load(archive, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["ball_vis"], positions.usable)
        np.testing.assert_array_equal(saved["ball_3d"], ball.trajectory.positions)
        np.testing.assert_array_equal(saved["ball_3d_valid"], ball.trajectory.valid)
        np.testing.assert_array_equal(
            saved["ball_rejection_code"], ball.trajectory.reasons
        )
        assert int(saved["ball_3d_valid"].sum()) == result["valid_3d"]
        if original is None:
            assert saved["player_valid"].shape == (0, result["frames"])
            assert saved["player_rejection_code"].shape == (0, result["frames"])
            assert (
                metadata["player_reconstruction"]["reason_code"]
                == result["player_reason"]
            )
            assert result["geometry"] == SHARED_GEOMETRY
            assert (
                ball_metadata["geometry_provenance"]["player_validation"] == "not_run"
            )
    unchanged_arrays = 0
    if original is not None:
        assert receipt.before.marker() is not None
        assert (root / before["exports"]["scene"]["files"]["scene"]["path"]).is_file()
        assert result["geometry"] == SAVED_GEOMETRY
        assert (
            ball_metadata["camera_geometry"] == before["artifacts"]["camera_alignment"]
        )
        old, new = (
            read_component_descriptor(
                root, item, node="scene_assembly", source_sha256=index["source_sha256"]
            )
            for item in (original, ref)
        )
        for field, encoded in old["payload"].items():
            if field in BALL_FIELDS or field == "metadata":
                continue
            if isinstance(encoded, dict) and set(encoded) == {"array"}:
                assert (
                    old["arrays"][encoded["array"]]
                    == new["arrays"][new["payload"][field]["array"]]
                )
                unchanged_arrays += 1
            else:
                assert new["payload"][field] == encoded
    row = {
        "clip": clip,
        "frames": result["frames"],
        "valid_3d": result["valid_3d"],
        "valid_percent": round(100 * result["valid_rate"], 5),
        "geometry": result["geometry"],
        "player_reason": result["player_reason"],
        "fixed_scene_arrays": unchanged_arrays,
        "previous_scene_id": None if original is None else original["artifact_id"],
        "previous_scene_descriptor_sha256": None
        if original is None
        else original["sha256"],
        "new_scene_id": ref["artifact_id"],
        "new_scene_descriptor_sha256": ref["sha256"],
        "new_archive_sha256": index["exports"]["scene"]["files"]["scene"]["sha256"],
        "external_json_sha256": positions.sha256,
        "prepare_seconds": round(result["prepare_seconds"], 5),
        "stage_seconds": round(result["stage_seconds"], 5),
        "export_seconds": round(result["export_seconds"], 5),
        "publish_seconds": round(result["elapsed_seconds"], 5),
    }
    for label, code in REJECTION_CODES.items():
        total = int((ball.trajectory.reasons == code).sum())
        assert total == result["ball_rejection"][label]
        row[f"reason_{label}"] = total
    views = positions.usable.sum(0)
    for count in range(5):
        at_count = views == count
        row[f"frames_{count}_views"] = int(at_count.sum())
        row[f"valid_{count}_views"] = int((at_count & ball.trajectory.valid).sum())
        assert result["view_count"][str(count)] == {
            "frames": row[f"frames_{count}_views"],
            "valid_3d": row[f"valid_{count}_views"],
        }
    for label, counts in result["point_status"].items():
        for field, count in counts.items():
            row[f"{label}_{field}"] = count
    for label, value in result["reprojection_px"].items():
        row[f"reprojection_{label}_px"] = round(value, 5)
    for stage, digests in result["pointer_digests"].items():
        for pointer, digest in digests.items():
            row[f"{stage}_{pointer}"] = digest
    return row, ball.trajectory.reprojection_px[ball.trajectory.inliers]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("smoke", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    smoke, full = (json.loads(path.read_text()) for path in (args.smoke, args.full))
    for key in ("dataset_sha256", "code_sha256", "side_digest"):
        assert smoke[key] == full[key]
    assert smoke["dataset_unchanged"] and full["dataset_unchanged"]
    first_pass = {row["clip_id"]: row for row in smoke["clips"]}
    rows, errors = [], []
    for result in full["clips"]:
        assert result["status"] == "published"
        if result["reused_publication"]:
            earlier = first_pass[result["clip_id"]]
            assert result["scene_assembly"] == earlier["scene_assembly"]
            assert result["pointer_digests"] == earlier["pointer_digests"]
            result = earlier
        row, error = verify_clip(args.dataset.resolve(), result)
        rows.append(row)
        errors.append(error)
    assert [row["clip"] for row in rows] == [f"clip_{i:03d}" for i in range(69)]
    counts = Counter(row["geometry"] for row in rows)
    assert counts == {SAVED_GEOMETRY: 55, SHARED_GEOMETRY: 14}
    assert sum(row["player_reason"] == "court_side_failed" for row in rows) == 1
    assert (
        sum(row["player_reason"] == "player_association_failed" for row in rows) == 13
    )

    def total(key):
        return sum(row[key] for row in rows)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "clips.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    quantiles = np.percentile(np.concatenate(errors), [50, 95, 100])
    summary = {
        "side_digest": full["side_digest"],
        "code_sha256": full["code_sha256"],
        "dataset_sha256": full["dataset_sha256"],
        "external_json_set_sha256": document_digest(
            {row["clip"]: row["external_json_sha256"] for row in rows}
        ),
        "clips": len(rows),
        "frames": total("frames"),
        "valid_3d": total("valid_3d"),
        "valid_rate": total("valid_3d") / total("frames"),
        "geometry": dict(counts),
        "fixed_scene_arrays": total("fixed_scene_arrays"),
        "ball_rejection": {
            label: total(f"reason_{label}") for label in REJECTION_CODES
        },
        "point_status": {
            label: {
                kind: total(f"{label}_{kind}")
                for kind in ("total", "out_of_frame", "used_in_3d")
            }
            for label in STATUSES
        },
        "view_count": {
            str(count): {
                "frames": total(f"frames_{count}_views"),
                "valid_3d": total(f"valid_{count}_views"),
            }
            for count in range(5)
        },
        "reprojection_px": dict(
            zip(("p50", "p95", "max"), quantiles.tolist(), strict=True)
        ),
        "seconds": {
            "smoke_wall": smoke["wall_seconds"],
            "full_wall": full["wall_seconds"],
            "sum_clip_publish": total("publish_seconds"),
            "independent_readback": time.perf_counter() - started,
        },
    }
    write_json_atomic(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
