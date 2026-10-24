"""Promote small evidence, selected cases and a local browser review index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import subprocess
from collections import Counter
from pathlib import Path

import analyze_probe as analyze
import plan_probe as plan
from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", required=True, type=Path)
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--evidence-output", type=Path)
    args = parser.parse_args()
    output = args.plan_root.resolve(strict=True)
    source_bundle = Path(__file__).resolve().parent
    bundle = (args.evidence_output or output / "evidence").resolve()
    replay = args.replay_root.resolve(strict=True)
    root = args.scenes_root.resolve(strict=True)
    if bundle.is_relative_to(root):
        raise ValueError("Evidence output must be outside the read-only input root")
    bundle.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((output / "inputs.json").read_text())
    verified = []
    for sid, inventory in inputs.items():
        for name, digest in inventory["hashes"].items():
            assert sha256(root / sid / name) == digest, (sid, name)
            verified.append(f"{sid}/{name}")
    replay_files = ["inputs.json", "policy.json", "smoke-1.json", "smoke-4.json"]
    for sid in ["B00", "B01", "B02", "B03"]:
        for condition in plan.CONDITIONS:
            replay_files += [
                f"{sid}/{condition}/{name}" for name in ["plan.json", "request.json"]
            ]
    for name in replay_files:
        assert sha256(output / name) == sha256(replay / name), name
    plan.dump(
        bundle / "input-validation.json",
        {
            "inputs_unchanged": True,
            "verified_inputs": verified,
            "replayed_plan_and_request_files": len(replay_files),
            "replay_byte_equal": True,
        },
    )
    for filename in [
        "summary.json",
        "inputs.json",
        "policy.json",
        "motion.json",
        "pair-checks.json",
        "render-timings.json",
        "smoke-1-timing.json",
        "smoke-4-timing.json",
    ]:
        (bundle / filename).write_bytes((output / filename).read_bytes())
    rows = []
    for sid in ["B00", "B01", "B02", "B03"]:
        for condition in plan.CONDITIONS:
            rows += json.loads((output / sid / condition / "qa.json").read_text())
    smoke_scene = plan.prior.load_scene(root, "B00")
    smoke_plan = {
        r["id"]: r
        for r in json.loads((output / "B00/recommended/plan.json").read_text())
    }
    capacity_qa = []
    for batch in [1, 4]:
        request = json.loads((output / f"smoke-{batch}.json").read_text())
        for camera in request["cameras"]:
            record = smoke_plan[camera["camera_id"]]
            directory = output / f"smoke-{batch}" / record["id"]
            arrays = analyze.NHTRenderArrays(
                **{
                    name: analyze.np.load(directory / f"{name}.npy", allow_pickle=False)
                    for name in ["rgb", "alpha", "depth"]
                }
            )
            values = analyze.diagnostics(record, arrays, smoke_scene)
            reference = analyze.load_arrays(output / "B00/recommended", record)
            values["capacity_batch"] = batch
            values["max_abs_difference_from_main"] = {
                name: float(
                    analyze.np.abs(
                        getattr(arrays, name) - getattr(reference, name)
                    ).max()
                )
                for name in ["rgb", "alpha", "depth"]
            }
            capacity_qa.append(values)
    plan.dump(bundle / "capacity-qa.json", capacity_qa)
    # One compact row per scene/condition/stratum; full raw summaries stay local.
    compact = []
    for entry in json.loads((output / "strata.json").read_text()):
        key = entry["stratum"]
        row = {
            "scene": entry["scene"],
            "condition": entry["condition"],
            "stratum": key,
            "value": entry[key],
            "n": entry["n"],
        }
        for gate in [
            "existing_gate",
            "candidate_target4",
            "candidate_roi01",
            "candidate_roi05",
            "candidate_roi10",
        ]:
            row[gate + "_pass"] = entry[gate + "_pass"]
        for key in [
            "all_joint_hole",
            "roi_joint_hole",
            "roi_alpha_lt09",
            "geometry_target_kp",
            "renderer_target_kp",
            "height_m",
            "hull_clearance_m",
            "line_ridge_distance_median_px",
        ]:
            row[key + "_median"] = (
                entry[key]["median"] if entry[key] is not None else None
            )
        row["coverage_counts"] = json.dumps(entry["coverage_counts"], sort_keys=True)
        compact.append(row)
    with (bundle / "strata.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(compact)
    # Preserve the changed distributions after candidate gates explicitly.
    accepted = []
    gates = [
        "existing_gate",
        "candidate_target4",
        "candidate_roi01",
        "candidate_roi05",
        "candidate_roi10",
    ]
    for threshold in [0.01, 0.05, 0.10, 0.25]:
        gate = f"candidate_alpha90_hole{int(threshold * 100):02d}"
        gates.append(gate)
        for row in rows:
            value = row["roi_alpha_lt09"]
            row[gate] = (
                row["candidate_target4"] and value is not None and value <= threshold
            )
    for sid in ["B00", "B01", "B02", "B03"]:
        for condition in plan.CONDITIONS:
            group = [
                r for r in rows if r["scene"] == sid and r["condition"] == condition
            ]
            for gate in gates:
                kept = [r for r in group if r[gate]]
                accepted.append(
                    {
                        "scene": sid,
                        "condition": condition,
                        "gate": gate,
                        "candidates": len(group),
                        "kept": len(kept),
                        "rejected": len(group) - len(kept),
                        "coverage": dict(Counter(r["coverage"] for r in kept)),
                        "hfov": dict(Counter(str(r["hfov"]) for r in kept)),
                        "shape": dict(Counter(r["shape"] for r in kept)),
                        "target": dict(Counter(r["target"] for r in kept)),
                    }
                )
    plan.dump(bundle / "gate-distributions.json", accepted)
    cases = json.loads((source_bundle / "cases.json").read_text())
    panels = []
    for item in cases:
        records = json.loads(
            (output / item["scene"] / item["condition"] / "plan.json").read_text()
        )
        record = next(
            r
            for r in records
            if r["kind"] == item["kind"] and r["index"] == item["index"]
        )
        metric = next(
            r
            for r in rows
            if r["scene"] == item["scene"]
            and r["condition"] == item["condition"]
            and r["id"] == record["id"]
        )
        item["camera_id"] = record["id"]
        item["metrics"] = {
            key: metric[key]
            for key in [
                "geometry_target_kp",
                "renderer_target_kp",
                "roi_joint_hole",
                "existing_gate",
                "candidate_target4",
                "line_ridge_distance_median_px",
            ]
        }
        array = analyze.load_arrays(output / item["scene"] / item["condition"], record)
        image = analyze.panel(record, array, metric, 480)
        captioned = Image.new("RGB", (image.width, image.height + 24), "#151a22")
        captioned.paste(image, (0, 24))
        ImageDraw.Draw(captioned).text((8, 5), item["label"], fill="#ffd26c")
        panels.append(captioned)
    full = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)))
    offset = 0
    for image in panels:
        full.paste(image, (0, offset))
        offset += image.height
    full.save(output / "review/representative-cases.jpg", quality=92)
    for category in ["clearer", "failure"]:
        selected = [
            p.resize((640, round(p.height * 640 / p.width)))
            for p, c in zip(panels, cases, strict=True)
            if c["group"] == category
        ]
        small = Image.new("RGB", (640, sum(p.height for p in selected)))
        offset = 0
        for image in selected:
            small.paste(image, (0, offset))
            offset += image.height
        small.convert("P", palette=Image.Palette.ADAPTIVE, colors=128).save(
            bundle / f"cases-{category}.png", optimize=True
        )
    plan.dump(bundle / "cases.json", cases)
    videos = json.loads((output / "videos.json").read_text())
    description = {
        "recommended-rgb-overlay": "推奨条件の連続96点：RGBと教師overlay",
        "compare-hull5": "D2-a：上段0%拡張、下段5%拡張。同じ弧長割合で比較",
        "compare-hfov45_90": "D2-b：上段92.5度、下段67.5度。同一位置・姿勢で比較",
        "compare-jitter0": "D6：上段1m独立jitter、下段0m。同一位置・Kで比較",
    }
    body = [
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>Court 新規視点品質比較</title>',
        "<style>body{font:16px system-ui;background:#141a23;color:#eef1f5;max-width:1200px;margin:2em auto}a{color:#86c4ff}video,img{max-width:100%;height:auto}section{padding:1em;border-bottom:1px solid #506070}table{border-collapse:collapse}td,th{padding:.5em;border:1px solid #506070}</style>",
        "<h1>Court 新規視点の品質比較</h1><p>全て新規NHT描画。採用前の診断用標本で、母集団の採択率を推定するものではありません。動画は12fpsで弧長順を再生しています。カメラの物理速度や時間軌道の定義ではありません。</p>",
        "<p>各パネルの左はRGB、右は教師overlay。緑=コート線、cyan/red=renderer支持のある/ないKP、黄=注視点、magenta=alpha&lt;.01またはdepth≤0。穴率0でもぼけ・線の不一致が残ります。</p>",
        '<p><a href="representative-cases.jpg">良例と失敗例の一覧</a> / <a href="../frames.csv">全frame QA</a> / <a href="../summary.json">シーン・条件別分布</a> / <a href="../strata.json">層別分布</a></p>',
    ]
    for sid in ["B00", "B01", "B02", "B03"]:
        body.append(f"<h2>{sid}</h2>")
        for suffix, title in description.items():
            filename = f"{sid}-{suffix}.mp4"
            body.append(
                f'<section><h3>{html.escape(title)}</h3><video controls preload="metadata" src="{filename}"></video><p><a href="{filename}">{filename}</a></p></section>'
            )
        for condition in plan.CONDITIONS:
            filename = f"{sid}-{condition}-contact.jpg"
            body.append(
                f'<details><summary>{condition} contact sheet</summary><a href="{filename}"><img loading="lazy" src="{filename}"></a></details>'
            )
    body.append("</html>")
    (output / "review/index.html").write_text("\n".join(body))
    artifacts = []
    for path in sorted((output / "review").iterdir()):
        artifacts.append(
            {
                "path": str(path.relative_to(Path.cwd())),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    plan.dump(
        bundle / "media-manifest.json", {"artifacts": artifacts, "videos": videos}
    )
    plan.dump(
        bundle / "local-evidence.json",
        {
            name: {
                "path": str(output / name),
                "bytes": (output / name).stat().st_size,
                "sha256": sha256(output / name),
            }
            for name in [
                "frames.csv",
                "strata.json",
                "summary.json",
                "analysis-time.txt",
            ]
        },
    )
    print(
        "verified inputs",
        len(verified),
        "replay files",
        len(replay_files),
        "media",
        len(artifacts),
        flush=True,
    )
    # Check the media as complete decodable files, beyond container metadata.
    for video in videos:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "2",
                "-i",
                video["path"],
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
    print("all videos fully decoded", flush=True)


if __name__ == "__main__":
    main()
