"""Produce small reviewable correspondence tables from completed, retained reports."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    selection = json.loads((bundle / "selection-reviewed.json").read_bytes())
    report = json.loads((args.reports / "full-publication.json").read_bytes())
    small = json.loads((args.reports / "small-relocated-publication.json").read_bytes())
    assert report["publication"]["replay_equal"] is True
    assert small["publication"]["replay_equal"] is True
    assert report["selected"] == selection["decisions"]["convert"]
    by_key = {e["key"]: e for e in selection["entries"]}
    assert {e["key"] for e in report["entries"]} == {e["key"] for e in selection["entries"] if e["decision"] == "convert"}
    snapshot_path = args.source_root / "data" / report["publication"]["snapshot"]["path"]
    snapshot = json.loads(snapshot_path.read_bytes())
    shards = [args.source_root / "data" / c["shard"]["path"] for c in snapshot["clips"]]
    unique = {(p.stat().st_dev, p.stat().st_ino): p.stat().st_size for p in shards}
    assert len(unique) == len(shards)
    summary = dict(
        schema_version=1, execution_code_commit=report["code_commit"],
        selection_sha256=report["selection_sha256"],
        inventoried_cameras=len(selection["entries"]), inventoried_frames=sum(e["frames"] for e in selection["entries"]),
        migrated_cameras=report["selected"], migrated_frames=report["publication"]["frames"],
        held_cameras=selection["decisions"]["hold"],
        held_frames=sum(e["frames"] for e in selection["entries"] if e["decision"] == "hold"),
        hold_reason_counts=selection["hold_reason_counts"],
        counts=report["publication"]["counts"], split_frames=report["publication"]["split_frames"],
        generated_negative_candidates=0, generated_unprocessed=0,
        receipt=report["publication"]["receipt"], snapshot=report["publication"]["snapshot"],
        replay_equal=True, small_replay_equal=True,
        scene_clips=len(report["scene_imports"]),
        scene_camera_artifacts=sum(len(c["references"]) for c in report["scene_imports"]),
        learning_samples=report["publication"]["learning_samples"],
        native_jpeg_bytes=sum(unique.values()), native_jpeg_archives=len(unique),
        legacy_unchanged=report["legacy_unchanged"],
        reports_root="outputs/ball_detection/migrate/video-v3/20261113",
    )
    assert summary["counts"] == selection["selected_counts"]
    assert summary["migrated_frames"] + summary["held_frames"] == summary["inventoried_frames"]
    fields = ["key", "original_path", "legacy_group", "recording_id", "camera", "split", "frames", "observed", "interpolated", "occlusion_estimated", "unresolved",
              "original_sha256", "annotation_path", "annotation_digest", "input_path", "migration_path", "source_media_path", "scene_artifact_path"]
    scenes = {clip["clip"]: clip["references"] for clip in report["scene_imports"]}
    with (bundle / "correspondence.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for result in report["entries"]:
            entry = by_key[result["key"]]
            refs = result["references"]
            scene = scenes[entry["clip"]][f"ball_detection/{entry['camera']}"]
            writer.writerow(dict(
                **{k: entry[k] for k in ("key", "legacy_group", "recording_id", "camera", "split", "frames", "original_sha256")},
                original_path=f"{selection['legacy_root']}/{entry['clip']}/outsource/{entry['camera']}_annotations.json",
                **entry["counts"], annotation_path=refs["annotation"]["path"], annotation_digest=refs["annotation"]["digest"],
                input_path=refs["expected_input"]["path"], migration_path=refs["migration"]["path"],
                source_media_path=refs["source_media"]["path"],
                scene_artifact_path=f"{summary['reports_root']}/meiji-active-ball-v3-20261113/scene-imports/{entry['clip']}/{scene['path']}",
            ))
    with (bundle / "held-inputs.csv").open("w", newline="") as output:
        fields = ["key", "camera", "split", "frames", "original_sha256", "reasons", "target_pointer", "target_declaration", "processing_evidence"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for entry in selection["entries"]:
            if entry["decision"] != "hold":
                continue
            target = entry.get("target_conflict_evidence", {})
            writer.writerow(dict(**{k: entry[k] for k in ("key", "camera", "split", "frames", "original_sha256")},
                                 reasons=";".join(entry["reasons"]), target_pointer=target.get("pointer", ""),
                                 target_declaration=target.get("text", ""), processing_evidence=entry["processing_evidence"]))
    counts = Counter()
    for entry in report["entries"]:
        counts.update({k: entry["summary"][k] for k in ("processed", "unprocessed", "position_usable", "final_unknown", "active_ball_negative_candidate")})
    assert counts["unprocessed"] == counts["active_ball_negative_candidate"] == 0
    summary["usage"] = dict(counts)
    (bundle / "results-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
