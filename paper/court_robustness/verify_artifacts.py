"""Verify the committed evidence; optionally bind it to the local source owners."""

from __future__ import annotations

import argparse
import json
import re
import subprocess

import numpy as np
from common import REPO, ROOT, sha256, sources, write_json
from make_scene_figures import SELECTION, render_overlay, validate_projection
from PIL import Image


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-local-sources", action="store_true")
    args = parser.parse_args()
    records = sources()
    require(len(records) == 4, "Paper must include all four supplied images")
    metadata = json.loads((ROOT / "evidence/inference_both.json").read_text())
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
    external = json.loads((ROOT / "evidence/external_figures.json").read_text())
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
    require(pages == 5, "Unexpected page overflow")
    log = (ROOT / "report.log").read_text()
    require(
        not any(
            term in log
            for term in (
                "Missing character",
                "Overfull",
                "undefined references",
                "undefined on input",
            )
        ),
        "LaTeX rendering/reference error",
    )
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
        "corpus_images_audited": 17256,
        "exact_matches": 0,
        "baseline_official_detections": counts,
        "baseline_control_error_px": error,
        "max_scene_label_reprojection_difference_px": max_projection_difference,
        "pdf_sha256": sha256(ROOT / "report.pdf"),
        "scope": "Artifact integrity and rendering/inference consistency, not accuracy against human ground truth.",
    }
    write_json(ROOT / "evidence/validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
