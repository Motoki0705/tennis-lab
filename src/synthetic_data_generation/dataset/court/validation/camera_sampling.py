"""Validate camera-sampling support and reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

PROBE_SCHEMA = "court_novel_view_probe_v1"
REPORT_SCHEMA = "court_novel_view_p6_acceptance_v1"
REQUIRED_COMMITS = {
    "5b4d4f64608ec8077222c52fdf814d40acc10bc1",
    "9471c8698077f0edac9e749208db9ef987cb5ca8",
    "b74732812b295189f230a192418375f56cec3bd6",
    "ebf2f3e75fd6c5dfc8c9d0b533800daaf17bd95f",
    "50e0e3c70c775e89333256213363badbf074f29d",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _load_probe(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != PROBE_SCHEMA
        or raw.get("status") != "passed"
        or raw.get("p6_sampling_probe_passed") is not True
    ):
        raise ValueError(f"Invalid probe: {path}")
    fingerprint = raw.get("content_fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError(f"Missing probe fingerprint: {path}")
    unsigned = dict(raw)
    del unsigned["content_fingerprint"]
    if _canonical_sha256(unsigned) != fingerprint:
        raise ValueError(f"Probe fingerprint mismatch: {path}")
    return raw


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-repeat", type=Path, required=True)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    probe_path = args.probe / "manifest.json"
    repeat_path = args.probe_repeat / "manifest.json"
    probe = _load_probe(probe_path)
    repeat = _load_probe(repeat_path)
    if probe_path.read_bytes() != repeat_path.read_bytes() or probe != repeat:
        raise RuntimeError("Independent same-seed probes are not byte-identical.")

    research_text = args.research.read_text()
    pins = json.loads(args.pins.read_text())
    repositories = pins.get("repositories") if isinstance(pins, dict) else None
    if not isinstance(repositories, list):
        raise ValueError("Research pins are invalid.")
    pinned_commits = {
        item.get("commit") for item in repositories if isinstance(item, dict)
    }
    if pinned_commits != REQUIRED_COMMITS:
        raise ValueError("Research pins do not match the pre-registered set.")
    for commit in REQUIRED_COMMITS:
        if commit not in research_text:
            raise ValueError(f"Research record omits pinned commit {commit}.")
    for failure in (
        "Unclipped global MultiNeRF ellipse",
        "FisherRF expected-information proposal generation",
        "Independent random jitter as the final dataset",
        "Camera-position convex hull alone",
    ):
        if failure not in research_text:
            raise ValueError(f"Research record omits failed hypothesis: {failure}.")

    source = _mapping(probe.get("source"), name="source")
    research_ref = _mapping(source.get("research"), name="source.research")
    pins_ref = _mapping(source.get("pins"), name="source.pins")
    if research_ref.get("sha256") != _sha256(args.research):
        raise ValueError("Probe research hash differs from the current record.")
    if pins_ref.get("sha256") != _sha256(args.pins):
        raise ValueError("Probe pin hash differs from the current record.")

    thresholds = _mapping(probe.get("thresholds"), name="thresholds")
    expected_thresholds = {
        "translation_limit_m": 0.25,
        "rotation_limit_deg": 1.5,
        "support_score_limit": 1.0,
        "near_plane_m": 0.1,
        "min_camera_height_m": 1.2,
        "min_image_margin_px": 0.0,
        "min_line_keypoints_visible": 14,
        "collision_neighbor_rank": 8,
        "min_collision_clearance_m": 0.25,
    }
    if thresholds != expected_thresholds:
        raise ValueError("Probe thresholds differ from pre-registration.")

    metrics = _mapping(probe.get("metrics"), name="metrics")
    extrapolation = _mapping(
        metrics.get("extrapolation_score"),
        name="metrics.extrapolation_score",
    )
    collision = _mapping(
        metrics.get("collision_clearance_m"),
        name="metrics.collision_clearance_m",
    )
    depth = _mapping(
        metrics.get("min_court_depth_m"),
        name="metrics.min_court_depth_m",
    )
    margin = _mapping(
        metrics.get("min_line_margin_px"),
        name="metrics.min_line_margin_px",
    )
    views = probe.get("views")
    passed = (
        metrics.get("safe_captured_anchor_count") == 42
        and metrics.get("selected_count") == 256
        and metrics.get("unique_selected_anchor_count") == 42
        and metrics.get("nontrivial_novel_pose_count") == 256
        and _number(
            metrics.get("accepted_candidate_fraction"),
            name="metrics.accepted_candidate_fraction",
        )
        >= 0.95
        and _number(
            metrics.get("selected_to_anchor_expansion_factor"),
            name="metrics.selected_to_anchor_expansion_factor",
        )
        >= 4.0
        and _number(
            metrics.get("selected_pairwise_minimum_pose_score"),
            name="metrics.selected_pairwise_minimum_pose_score",
        )
        >= 0.5
        and _number(
            extrapolation.get("maximum"),
            name="metrics.extrapolation_score.maximum",
        )
        <= 1.0
        and _number(
            collision.get("minimum"),
            name="metrics.collision_clearance_m.minimum",
        )
        >= 0.25
        and _number(
            depth.get("minimum"),
            name="metrics.min_court_depth_m.minimum",
        )
        > 0.1
        and _number(
            margin.get("minimum"),
            name="metrics.min_line_margin_px.minimum",
        )
        >= 0.0
        and isinstance(views, list)
        and len(views) == 256
    )
    if not passed:
        raise RuntimeError(f"P6 gates failed: {metrics}")

    unsigned = {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "p6_complete": True,
        "method": (
            "captured-pose local SE(3) support ball, explicit geometry gates, "
            "and independent farthest-view selection"
        ),
        "probe": {
            "manifest_sha256": _sha256(probe_path),
            "content_fingerprint": probe["content_fingerprint"],
            "repeat_byte_identical": True,
        },
        "research": {
            "sha256": _sha256(args.research),
            "pins_sha256": _sha256(args.pins),
            "pinned_commits": sorted(REQUIRED_COMMITS),
        },
        "thresholds": thresholds,
        "metrics": metrics,
    }
    report = dict(unsigned)
    report["content_fingerprint"] = _canonical_sha256(unsigned)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            temporary.rmdir()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
