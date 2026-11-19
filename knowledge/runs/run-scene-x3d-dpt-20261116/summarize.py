"""Curate immutable raw results and fully decode the comparison video streams.

CPU only. Raw reports, stores, arrays and videos remain in outputs; this writes
small summaries with hashes and preserves every failing validation frame.
"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(path):
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha(path))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    evidence = args.evidence
    names = {
        "validation-parity-eager.json": "validation-parity.json",
        "validation-parity-compiled.json": "validation-parity-compiled.json",
        "validation-parity-torch214.json": "validation-parity-torch214.json",
        "validation-parity-gpu-decode.json": "validation-parity-gpu-decode.json",
        "validation-paired-source.json": "validation-paired-source.json",
    }
    for destination, original in names.items():
        path = root / original
        data = json.loads(path.read_text())
        data["raw_report"] = receipt(path)
        data["curation"] = "Every mismatch retained. Per-frame passing differences remain in the immutable raw report."
        for row in data["by_precision"].values():
            differences = row.pop("all_differences")
            if row["frames"] != len(differences):
                raise ValueError("Frame count differs from raw report")
            if row["mismatches"] != [item for item in differences if not item["passed"]]:
                raise ValueError("Mismatch list differs from raw report")
            row["mismatch_count"] = len(row["mismatches"])
        write(evidence / destination, data)

    metrics = dict(
        schema="scene_dpt_comparison_summary.v1",
        scope="Two completed Meiji clips. No ground truth; reprojection is consistency, not accuracy.",
        timing_scope="Ball component wall time includes load, IO/resize, inference, candidates, gates and unload. Separate cam0 probe has populated torch.compile cache. No pure forward latency claim.",
        gpu_reservation="shared training queue, resource=all, per-process allocator fraction=0.70",
        primary_four_camera={}, canonical_three_camera={}, fixed_geometry_diagnostic={},
    )
    for directory, key in (("meiji-four-camera-final", "primary_four_camera"),
                           ("meiji-three-camera-final", "canonical_three_camera")):
        for clip in ("clip_000", "clip_018"):
            variants = {}
            for variant in ("i935", "x3d"):
                path = root / directory / variant / clip / "metrics.json"
                raw = json.loads(path.read_text())
                record = {name: raw[name] for name in (
                    "commit", "checkpoint_sha256", "source_marker_sha256", "status", "num_frames", "fps",
                    "camera_ids", "ball_detection_frames", "ball_points_frames", "any_camera_2d_frames",
                    "ball_3d_valid_frames", "reprojection_mean_px", "ball_probe", "ball_stage_seconds", "torch", "gpu",
                )}
                record["failure_reason"] = raw["failure"]["reason"] if raw["failure"] else None
                record["raw_metrics"] = receipt(path)
                record["all_camera_ball_component_seconds"] = sum(raw["ball_stage_seconds"].values())
                record["stage_status"] = raw["receipt"]["stage_status"]
                variants[variant] = record
            metrics[key][clip] = variants
    for variant in ("i935", "x3d"):
        path = root / "meiji-fixed-geometry" / variant / "clip_000/metrics.json"
        raw = json.loads(path.read_text())
        metrics["fixed_geometry_diagnostic"][variant] = {
            name: raw[name] for name in (
                "comparison_scope", "canonical_pipeline_status", "canonical_pipeline_failure", "geometry_source",
                "ball_3d_valid_frames", "reprojection_inlier_count", "reprojection_mean_px", "reprojection_median_px",
                "reprojection_p95_px", "replay_seconds",
            )
        }
        metrics["fixed_geometry_diagnostic"][variant]["raw_metrics"] = receipt(path)
    write(evidence / "meiji-comparison.json", metrics)

    videos = []
    for directory in ("meiji-four-camera-final", "meiji-fixed-geometry"):
        for path in sorted((root / directory).glob("*.mp4")):
            clip = path.name[:8]
            source_metrics = root / directory / "x3d" / clip / "metrics.json"
            expected_frames = json.loads(source_metrics.read_text())["num_frames"]
            result = subprocess.run([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames,codec_name", "-of", "json", str(path),
            ], check=True, capture_output=True, text=True)
            if result.stderr.strip():
                raise ValueError(f"Video decoder reported an error: {path}: {result.stderr}")
            streams = json.loads(result.stdout)["streams"]
            if len(streams) != 1:
                raise ValueError(f"Unexpected video stream count: {path}")
            stream = streams[0]
            if int(stream["nb_read_frames"]) != expected_frames:
                raise ValueError(f"Incomplete comparison video: {path}")
            if Fraction(stream["avg_frame_rate"]) != Fraction(60000, 1001):
                raise ValueError(f"Comparison FPS differs from source: {path}")
            videos.append(dict(**receipt(path), **stream, full_stream_decode="ffprobe -count_frames, exit 0, no decoder errors"))
    if len(videos) != 11:
        raise ValueError(f"Expected 10 four-camera comparisons and one fixed-geometry comparison, found {len(videos)}")
    write(evidence / "media-manifest.json", dict(schema="scene_dpt_media.v1", videos=videos))
    print(json.dumps(dict(parity_reports=len(names), comparison_clips=2, videos=len(videos))), flush=True)


if __name__ == "__main__":
    main()
