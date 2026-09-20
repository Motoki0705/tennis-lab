"""Verify the committed evidence; optionally bind it to the local source owners."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from alignment_evidence import BUNDLE, read_bundle, validate_geometry
from build_paper import source_digests
from common import PAPER_PAGES, REPO, ROOT, sha256, sources, write_json
from drift_evidence import BUNDLE as DRIFT_BUNDLE
from drift_evidence import validate as validate_drift_geometry
from homography_evidence import read_results as read_homographies
from homography_evidence import table_text
from make_drift_figure import drift_table
from make_scene_figures import SELECTION, render_overlay, validate_projection
from PIL import Image


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_build_receipt(root: Path) -> None:
    receipt = json.loads((root / "evidence/build.json").read_text())
    require(receipt["schema"] == "court_paper_build_v1", "Unknown build receipt")
    require(
        receipt["source_sha256"] == source_digests(root),
        "PDF sources differ from the recorded build",
    )
    require(
        receipt["pdf_sha256"] == sha256(root / "report.pdf"),
        "PDF differs from the recorded build",
    )
    require(
        receipt["layout_glyph_reference_checks"] == "passed",
        "Build validation did not pass",
    )


def validate_local_weights(metadata: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from omegaconf import OmegaConf

    checkpoint_path = Path(metadata["checkpoint"])
    require(
        sha256(checkpoint_path) == metadata["ours_sha256"],
        "Checkpoint SHA-256 differs from inference",
    )
    require(
        sha256(Path(metadata["baseline_checkpoint"])) == metadata["baseline_sha256"],
        "TCD weight SHA-256 differs from inference",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = json.loads((ROOT / "evidence/checkpoint_config.json").read_text())
    saved_bundle = json.loads((ROOT / "evidence/target_bundle.json").read_text())
    require(
        OmegaConf.to_container(checkpoint["hyper_parameters"]["config"], resolve=True)
        == saved_config,
        "Checkpoint configuration differs from the recorded configuration",
    )
    require(
        checkpoint["hyper_parameters"]["target_bundle_state"] == saved_bundle,
        "Checkpoint target bundle differs from the recorded contract",
    )
    require(
        checkpoint["epoch"] == metadata["epoch"]
        and checkpoint["global_step"] == metadata["global_step"],
        "Checkpoint training step differs",
    )
    require(
        not torch.cuda.is_initialized(),
        "Source validation unexpectedly initialized CUDA",
    )


def validate_alignment_method(*, check_local_sources: bool = False) -> dict:
    figures = json.loads((ROOT / "evidence/alignment_figures.json").read_text())
    require(
        sha256(BUNDLE / "manifest.json") == figures["bundle_manifest_sha256"],
        "Changed method evidence manifest",
    )
    manifest, arrays = read_bundle()
    actual = validate_geometry(manifest, arrays)
    require(actual == figures["verification"], "Method geometry verification differs")
    require(
        sha256(ROOT / "evidence/scene_sources/B00.json")
        == figures["display"]["method_projection"]["alignment_bundle_sha256"],
        "Changed final B00 alignment for method figure",
    )
    for name, digest in figures["figures"].items():
        require(sha256(ROOT / "figures" / name) == digest, "Changed method figure")
        require(
            name in (ROOT / "report.tex").read_text(), "Method figure absent from paper"
        )
    if check_local_sources:
        for path, digest in manifest["source_files"].items():
            require(
                sha256(REPO / path) == digest,
                f"Changed alignment method source: {path}",
            )
        scene = REPO / "data/synthetic_data_generation/scenes/B00"
        require(
            np.array_equal(
                arrays["points_xyzrgb"],
                np.load(scene / "reconstruction/export/points_scene.npy"),
            ),
            "Bundled point cloud differs from source",
        )
        with np.load(scene / "alignment/line-heatmaps/heatmaps.npz") as original:
            for key in (
                "camera_ids",
                "included_in_aggregate",
                "projected_offsets",
                "projected_points_uv",
                "projected_probabilities",
                "proximity_weights",
                "evidence_sum",
            ):
                require(
                    np.array_equal(arrays[key], original[key]),
                    f"Bundled {key} differs from source",
                )
            for camera_id in manifest["selection"]:
                i = original["camera_ids"].tolist().index(camera_id)
                start, stop = original["probability_offsets"][i : i + 2]
                probability = original["probability_values"][start:stop].reshape(
                    original["probability_shapes"][i]
                )
                require(
                    np.array_equal(arrays[f"probability_{camera_id}"], probability),
                    "Bundled LINE probability differs from source",
                )
                require(
                    sha256(BUNDLE / f"{camera_id}.png")
                    == sha256(
                        scene / "reconstruction/export/images" / f"{camera_id}.png"
                    ),
                    "Bundled method RGB differs from source",
                )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-local-sources", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    method_checks = validate_alignment_method(
        check_local_sources=args.check_local_sources
    )
    drift_checks = validate_drift_geometry(check_local_sources=args.check_local_sources)
    drift = json.loads((ROOT / "evidence/drift_figures.json").read_text())
    require(
        sha256(DRIFT_BUNDLE / "manifest.json") == drift["bundle_manifest_sha256"]
        and sha256(DRIFT_BUNDLE / "measurements.json") == drift["measurements_sha256"],
        "Changed SfM drift source bundle or measurements",
    )
    require(drift_checks == drift["verification"], "SfM drift verification differs")
    for name, digest in drift["tables"].items():
        require(sha256(ROOT / name) == digest, "Changed SfM drift table")
        require(
            (ROOT / name).read_text() == drift_table(),
            "SfM drift table differs from measurements",
        )
        require(
            name in (ROOT / "report.tex").read_text(), "SfM table absent from paper"
        )
    for name, digest in drift["figures"].items():
        require(sha256(ROOT / "figures" / name) == digest, "Changed SfM drift figure")
        require(
            name in (ROOT / "report.tex").read_text(),
            "SfM drift figure absent from paper",
        )
    records = sources()
    require(len(records) == 4, "Paper must include all four supplied images")
    metadata = json.loads((ROOT / "evidence/inference_both.json").read_text())
    for name, digest in metadata["configuration_sha256"].items():
        require(
            sha256(ROOT / "evidence" / name) == digest,
            "Changed inference configuration evidence",
        )
    if args.check_local_sources:
        validate_local_weights(metadata)
    audit = json.loads((ROOT / "evidence/dataset_audit.json").read_text())
    require(
        metadata["device"] == "cpu" and not metadata["cuda_initialized"],
        "CPU provenance",
    )
    require(
        metadata["ids"] == [r["id"] for r in records],
        "Inference subset differs from paper",
    )
    require(
        metadata["input_manifest_sha256"]
        == sha256(ROOT / "evidence/inputs.json")
        == audit["input_manifest_sha256"],
        "Changed input manifest",
    )
    counts = []
    for record in records:
        ident = record["id"]
        require(
            sha256(ROOT / record["paper_path"])
            == record["sha256"]
            == metadata["inputs"][ident],
            "Changed input photo",
        )
        if args.check_local_sources:
            require(
                sha256(REPO / record["supplied_path"]) == record["sha256"],
                "Changed supplied image",
            )
        for model in ("baseline", "ours"):
            path = ROOT / f"evidence/predictions/{ident}_{model}.npz"
            require(
                sha256(path) == metadata["prediction_sha256"][path.name],
                "Changed prediction array",
            )
            with np.load(path) as output:
                require(output["raw_kp"].shape == (14, 2), "Wrong KP shape")
                if model == "baseline":
                    probability = output["kp_probability"]
                    require(
                        probability.shape == (14, 360, 640), "Wrong TCD heatmap shape"
                    )
                    counts.append(int(np.isfinite(output["raw_kp"]).all(axis=1).sum()))
                    require(
                        not bool(output["homography_found"]),
                        "Paper's TCD H status is stale",
                    )
                else:
                    probability = output["line_probability"]
                    require(
                        tuple(probability.shape) == tuple(output["input_hw"]),
                        "LINE grid differs from network input",
                    )
                    require(
                        bool(output["homography_found"])
                        and np.isfinite(output["raw_kp"]).all(),
                        "Paper's model H status is stale",
                    )
                require(
                    np.isfinite(probability).all()
                    and probability.min() >= 0
                    and probability.max() <= 1,
                    "Invalid probability",
                )
    require(counts == [3, 1, 0, 1], "Paper's TCD counts differ from saved inference")
    require(
        sum(v["count"] for v in audit["corpora"].values()) == 17256,
        "Incomplete corpus audit",
    )
    require(
        all(
            not v["file_matches"] and not v["pixel_matches"]
            for v in audit["corpora"].values()
        ),
        "Duplicate input found",
    )
    with np.load(ROOT / "evidence/control_baseline.npz") as control:
        error = float(
            np.linalg.norm(control["aligned_kp"] - control["gt"], axis=1).mean()
        )
        require(
            np.isfinite(control["raw_kp"]).all()
            and abs(error - metadata["control_mean_error_px"]) < 1e-8
            and error < 2,
            "Failed TCD positive control",
        )
    scene_manifest = json.loads(
        (ROOT / "evidence/scene_visualization.json").read_text()
    )
    scene_figures = []
    max_projection_difference = 0.0
    for sid, expected_indices in SELECTION.items():
        path = ROOT / f"evidence/scene_sources/{sid}.json"
        bundle = json.loads(path.read_text())
        require(
            sha256(path) == scene_manifest[sid]["bundle_sha256"],
            "Changed scene geometry bundle",
        )
        require(
            [v["sample"]["sample_index"] for v in bundle["views"]] == expected_indices,
            "Changed scene selection",
        )
        require(
            len({v["sample"]["trajectory_group_id"] for v in bundle["views"]}) == 3,
            "Repeated trajectory group",
        )
        require(
            audit["synthetic_manifest_sha256"][sid] == bundle["dataset_sha256"],
            "Audited and illustrated dataset differ",
        )
        if args.check_local_sources:
            require(
                sha256(REPO / bundle["source_dataset"]) == bundle["dataset_sha256"],
                "Changed source dataset",
            )
            require(
                sha256(REPO / bundle["source_alignment"]) == bundle["alignment_sha256"],
                "Changed source alignment",
            )
        for view, generated in zip(
            bundle["views"], scene_manifest[sid]["views"], strict=True
        ):
            source = ROOT / view["bundled_rgb"]
            require(sha256(source) == view["rgb_sha256"], "Changed 3DGS RGB")
            if args.check_local_sources:
                require(
                    sha256(REPO / view["source_rgb"]) == view["rgb_sha256"],
                    "Different renderer source",
                )
            max_projection_difference = max(
                max_projection_difference, validate_projection(bundle, view["sample"])
            )
            path = ROOT / view["figure"]
            require(sha256(path) == generated["sha256"], "Changed overlay PNG")
            regenerated = render_overlay(bundle, view["sample"], Image.open(source))
            require(
                np.array_equal(np.asarray(regenerated), np.asarray(Image.open(path))),
                "Overlay not reproducible from real render + camera",
            )
            scene_figures.append(path.name)
    homographies = read_homographies()
    require(
        (ROOT / "tables/homography.tex").read_text() == table_text(homographies),
        "Confidence homography table differs",
    )
    require(
        "tables/homography.tex" in (ROOT / "report.tex").read_text(),
        "Confidence homography table absent from paper",
    )
    external = json.loads((ROOT / "evidence/external_figures.json").read_text())
    require(
        external["homography_evidence_sha256"]
        == sha256(ROOT / "evidence/homography/results.json"),
        "External figures use stale homography evidence",
    )
    for name, digest in external["figures"].items():
        require(
            sha256(ROOT / "figures" / name) == digest,
            "Changed external prediction panel",
        )
    tex = (ROOT / "report.tex").read_text()
    require(
        all(name in tex for name in scene_figures),
        "A 3DGS overlay is absent from the paper",
    )
    require(
        all("\\external{" + r["id"] + "}" in tex for r in records),
        "A supplied image is absent from the paper",
    )
    info = subprocess.check_output(["pdfinfo", str(ROOT / "report.pdf")], text=True)
    match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    require(match is not None, "No PDF page count")
    pages = int(match.group(1))
    require(pages == PAPER_PAGES, "Unexpected page overflow")
    validate_build_receipt(ROOT)
    text = subprocess.check_output(
        ["pdftotext", str(ROOT / "report.pdf"), "-"], text=True
    )
    require(
        "??" not in text and "Joliette" not in text and "Bay Harbor" not in text,
        "Stale text or broken references",
    )
    result = {
        "status": "passed",
        "pages": pages,
        "supplied_images": len(records),
        "scene_overlays": len(scene_figures),
        "overlays_per_scene": 3,
        "source_owner_check": args.check_local_sources,
        "local_weights_and_config_checked": args.check_local_sources,
        "build_receipt_verified": True,
        "alignment_method": method_checks,
        "sfm_temporal_ground_audit": drift_checks,
        "hybrid_homography": {
            key: {
                **{
                    field: item[field]
                    for field in (
                        "status",
                        "inlier_count",
                        "inlier_rms_px",
                        "threshold_px",
                        "candidate_count",
                        "refined_count",
                        "alternative_score_gap",
                    )
                },
                "stage_line_support": {
                    stage: {
                        field: value["line"][field]
                        for field in ("forward_support", "reverse_support")
                    }
                    for stage, value in item["stages"].items()
                    if value["line"] is not None
                },
            }
            for key, item in homographies["images"].items()
        },
        "corpus_images_audited": 17256,
        "exact_matches": 0,
        "baseline_official_detections": counts,
        "baseline_control_error_px": error,
        "max_scene_label_reprojection_difference_px": max_projection_difference,
        "pdf_sha256": sha256(ROOT / "report.pdf"),
        "scope": "Artifact integrity and rendering/inference consistency, not accuracy against human ground truth.",
    }
    if args.write_report:
        write_json(ROOT / "evidence/validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
