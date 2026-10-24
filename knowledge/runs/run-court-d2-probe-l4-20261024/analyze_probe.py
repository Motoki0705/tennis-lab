"""Verify recovered probe files, stratify QA and prepare unchanged-image review."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.synthetic_data_generation.dataset.court.scale.core import QualityPolicy, digest
from src.synthetic_data_generation.dataset.court.scale.quality import (
    quality_rejection_reasons,
    roi_mask,
    validate_quality_binding,
)
from src.utils.schema.court import COURT_SKELETON

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "run-court-d2-support-cpu-20261024"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def distribution(values: list[float]) -> dict[str, Any]:
    return (
        {
            "n": len(values),
            "min": min(values),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": max(values),
        }
        if values
        else {"n": 0}
    )


def summarize(frames: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(frames)
    return {
        "n": n,
        "accepted": sum(f["accepted"] for f in frames),
        "raw_acceptance": sum(f["accepted"] for f in frames) / n if n else None,
        "reasons": dict(Counter(r for f in frames for r in f["reasons"])),
        **{
            key: distribution([f[key] for f in frames if f[key] is not None])
            for key in (
                "roi_alpha_lt_0_9",
                "roi_hole_fraction",
                "target_visible_kp",
                "roi_gray_mean_0_255",
                "roi_laplacian_variance",
                "jpeg_bytes",
                "hull_clearance_m",
                "height_m",
            )
        },
    }


def overlay(rgb: Image.Image, record: dict[str, Any]) -> Image.Image:
    image = rgb.copy()
    draw = ImageDraw.Draw(image)
    target = record["target_court"]["binding"]["court_instance_id"]
    court = next(
        c for c in record["projection"]["courts"] if c["court_instance_id"] == target
    )
    points = {c["class_id"]: c["points"][0] for c in court["classes"]}
    for a, b in COURT_SKELETON:
        if a < 14 and b < 14 and points[a]["in_front"] and points[b]["in_front"]:
            draw.line([tuple(points[i]["uv"]) for i in (a, b)], fill="yellow", width=1)
    for i, p in points.items():
        if p["in_front"] and p["in_frame"]:
            x, y = p["uv"]
            color = "lime" if p["renderer_visible"] else "orange"
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.text(
                (x + 4, y + 2), str(i), fill=color, stroke_width=1, stroke_fill="black"
            )
    return image


def verify_bundle(root: Path) -> dict[str, Any]:
    manifest = read(root / "review-manifest.json")
    assert manifest["schema"] == "court_d2_review_bundle_v1"
    expected = {entry["path"] for entry in manifest["files"]}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert actual == expected | {"review-manifest.json"}
    for entry in manifest["files"]:
        path = root / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    state = read(root / "probe/state.json")
    assert state["source_commit"] == "4916def72d0b46728572006d7fbb19076460ec02"
    assert state["outcome"] == "completed"
    assert state["draw_attempts"] == state["completed_frames"] == 192
    selection = read(root / "probe/selection.json")
    assert isinstance(selection, dict)
    assert selection == read(FROZEN / "render-selection.json")
    assert (
        digest({k: v for k, v in selection.items() if k != "selection_sha256"})
        == selection["selection_sha256"]
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    args = parser.parse_args()
    selection = verify_bundle(args.recovered)
    quality = QualityPolicy(
        schema="court_scale_quality_policy_v2",
        rule_version="target_kp4_roi_holes_1pct_v1",
        minimum_target_visible_kp=4,
        maximum_roi_hole_fraction=0.01,
    )
    geometry = {
        f"{r['variant']}/{r['scene']}/{r['sample_id']}": r
        for r in read(args.geometry)["frames"]
    }
    args.review.mkdir(parents=True, exist_ok=False)
    frames = []
    scenarios = []
    pages = []
    for scenario in selection["scenarios"]:
        variant, scene = scenario["variant"], scenario["scene"]
        report = read(args.recovered / "probe/reports" / f"{variant}-{scene}.json")
        by_id = {r["sample_id"]: r for r in report["results"]}
        assert set(by_id) == {r["sample_id"] for r in scenario["samples"]}
        sheet = Image.new("RGB", (1280, 1260), "black")
        rows = []
        cards = []
        for i, selected in enumerate(scenario["samples"]):
            sample_id = selected["sample_id"]
            relative = f"{variant}/{scene}/{sample_id}"
            original = args.recovered / "probe/frames" / relative
            record = read(original / "record.json")
            validate_quality_binding(record["qa"], quality)
            assert record["reasons"] == quality_rejection_reasons(record["qa"], quality)
            result = by_id[sample_id]
            assert result["accepted"] == (not record["reasons"])
            assert result["reasons"] == record["reasons"]
            assert result["qa"] == record["qa"]
            assert result["framing"] == selected["framing"]
            source = geometry[relative]
            assert source["planned_record_sha256"] == selected["planned_record_sha256"]
            image = Image.open(original / "rgb.jpg").convert("RGB")
            assert image.size == (640, 360)
            gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            roi = roi_mask(record, 0.01)
            interior = cv2.erode(
                roi.astype(np.uint8), np.ones((3, 3), np.uint8)
            ).astype(bool)
            row = {
                "variant": variant,
                "scene": scene,
                "sample_id": sample_id,
                "framing": selected["framing"],
                "accepted": result["accepted"],
                "reasons": result["reasons"],
                **{
                    k: result["qa"][k]
                    for k in (
                        "roi_alpha_lt_0_9",
                        "roi_hole_fraction",
                        "target_visible_kp",
                    )
                },
                **{k: source[k] for k in ("height_m", "hull_clearance_m", "hfov_deg")},
                "jpeg_bytes": (original / "rgb.jpg").stat().st_size,
                "roi_gray_mean_0_255": float(gray[roi].mean()) if roi.any() else None,
                "roi_laplacian_variance": float(
                    cv2.Laplacian(gray, cv2.CV_64F)[interior].var()
                )
                if interior.any()
                else None,
            }
            rows.append(row)
            marked = overlay(image, record)
            output = args.review / variant / scene / sample_id
            output.mkdir(parents=True)
            (output / "rgb.jpg").write_bytes((original / "rgb.jpg").read_bytes())
            marked.save(output / "overlay.png")
            canvas = Image.new("RGB", (320, 420), "black")
            canvas.paste(image.resize((320, 180)), (0, 0))
            canvas.paste(marked.resize((320, 180)), (0, 180))
            draw = ImageDraw.Draw(canvas)
            alpha = row["roi_alpha_lt_0_9"]
            assert alpha is not None
            caption = f"{i + 1} {selected['framing']} {'PASS' if row['accepted'] else 'REJECT'} KP={row['target_visible_kp']}\nalpha<.9={alpha:.3f} hull={row['hull_clearance_m']:.2f}m\n{sample_id[:16]}"
            draw.multiline_text((4, 363), caption, fill="white")
            sheet.paste(canvas, ((i % 4) * 320, (i // 4) * 420))
            cards.append(
                f'<article><p>{html.escape(caption)}</p><img src="{relative}/rgb.jpg"><img src="{relative}/overlay.png"></article>'
            )
        page_name = f"{variant}-{scene}"
        sheet.save(args.review / f"{page_name}.png")
        pages.append(f'<li><a href="{page_name}.html">{page_name}</a></li>')
        (args.review / f"{page_name}.html").write_text(
            '<meta charset="utf-8"><style>body{background:#222;color:white}img{width:640px}article{margin:20px}</style>'
            + "".join(cards)
        )
        frames.extend(rows)
        weighted_accepted = weighted_jpeg = 0.0
        strata = {}
        for framing, plan in scenario["strata"].items():
            subset = [r for r in rows if r["framing"] == framing]
            assert len(subset) == plan["sampled"]
            assert (plan["geometry_population"] == 0) == (len(subset) == 0)
            factor = plan["geometry_population"] / len(subset) if subset else 0.0
            weighted_accepted += sum(r["accepted"] for r in subset) * factor
            weighted_jpeg += (
                sum(r["jpeg_bytes"] for r in subset if r["accepted"]) * factor
            )
            strata[framing] = {**summarize(subset), **plan}
        population = sum(s["geometry_population"] for s in scenario["strata"].values())
        assert (
            abs(
                weighted_accepted / population
                - report["weighted_acceptance_given_geometry"]
            )
            < 1e-12
        )
        scenarios.append(
            {
                "variant": variant,
                "scene": scene,
                **summarize(rows),
                "strata": strata,
                "geometry_population": population,
                "weighted_accepted_population": weighted_accepted,
                "weighted_acceptance_given_geometry": weighted_accepted / population,
                "weighted_accepted_jpeg_bytes": weighted_jpeg,
                "weighted_bytes_per_accepted_jpeg": weighted_jpeg / weighted_accepted
                if weighted_accepted
                else None,
                "render_qa_archive_seconds": report["render_qa_archive_seconds"],
                "memory": report["memory"],
            }
        )
    variants = {}
    for variant in selection["recipes"]:
        selected_scenarios = [r for r in scenarios if r["variant"] == variant]
        subset = [r for r in frames if r["variant"] == variant]
        accepted = sum(r["weighted_accepted_population"] for r in selected_scenarios)
        population = sum(r["geometry_population"] for r in selected_scenarios)
        variants[variant] = {
            **summarize(subset),
            "weighted_accepted_population": accepted,
            "geometry_population": population,
            "weighted_acceptance_given_geometry": accepted / population,
            "weighted_accepted_full_population": sum(
                r["strata"]["full"]["geometry_population"]
                * r["strata"]["full"]["raw_acceptance"]
                for r in selected_scenarios
                if r["strata"]["full"]["n"]
            ),
            "hull_outside": summarize([r for r in subset if r["hull_clearance_m"] < 0]),
            "hull_inside": summarize([r for r in subset if r["hull_clearance_m"] >= 0]),
        }
    write(
        args.output,
        {
            "schema": "court_d2_l4_analysis_v1",
            "selection_sha256": selection["selection_sha256"],
            "scenarios": scenarios,
            "variants": variants,
            "frames": frames,
            "notes": [
                "Balanced raw counts are not population rates; descriptive N_h/n_h weights use the frozen geometry population.",
                "Tiny correlated strata do not certify production acceptance; brightness and Laplacian variance are diagnostic, not new QA gates.",
                "All recovered JPEGs, including QA rejects, are reviewed without changing membership.",
            ],
        },
    )
    (args.review / "index.html").write_text(
        '<meta charset="utf-8"><h1>Frozen D2 probe: raw / overlay</h1><p>192 independent frozen views; no temporal trajectory implied.</p><ul>'
        + "".join(pages)
        + "</ul>"
    )
    print(
        json.dumps(
            {
                k: {
                    "raw": [v["accepted"], v["n"]],
                    "weighted_rate": v["weighted_acceptance_given_geometry"],
                }
                for k, v in variants.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
