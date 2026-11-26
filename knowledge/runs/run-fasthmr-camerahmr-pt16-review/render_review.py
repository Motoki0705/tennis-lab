"""Render synchronized four-camera and 3D comparisons from saved meshes."""

import json
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass
from html import escape
from pathlib import Path

import cv2
import numpy as np
from meshes_for_review import COHORT_SHA, SOURCE, cohort_records

from src.submodules.vendor.gvhmr.body_model import load_smpl_faces
from src.tennis_scene.evaluation.player_pose.bundle import pinned_file
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.io import write_json_atomic
from src.utils.rendering.mesh_renderer import MeshRenderer
from src.utils.schema.court import (
    COURT_SKELETON,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)
from src.utils.video import OpenCVVideoFrameReader, VideoWriter

MODELS = ("GVHMR (archive)", "FastHMR CameraHMR PT16")
FOUND_IN = ("gvhmr-archive", "pt16")
PALETTE = ((30, 193, 244), (238, 167, 87), (138, 226, 118), (210, 130, 246))
AXES = (
    "source_frames",
    "player_ids",
    "camera_ids",
    "selected_view_index",
    "segment_index",
    "selected_view_observed",
    "K",
    "R",
    "t",
)
GEOMETRY = (
    "camera_mesh_K",
    "camera_valid",
    "court_valid",
    "root_valid",
    "heading_valid",
    "placement_root_reason",
    "camera_vertices_m",
    "camera_coco17_m",
    "court_vertices_m",
    "court_coco17_m",
)
PRODUCTS = {
    "before": "配置前・選択ビュー",
    "after": "配置後・4カメラ",
    "joints": "配置後・COCO17関節",
    "3d": "配置前後・3D",
}
WIDTH, HEIGHT = 640, 360
INK = (240, 245, 248)


@dataclass(frozen=True)
class Comparison:
    clip: dict
    axes: dict
    predictions: tuple[dict, dict]
    digests: dict

    @classmethod
    def read(cls, clip, predictions):
        axes, arrays, hashes = None, [], {}
        for condition in FOUND_IN:
            summary = json.loads((predictions / f"review-{condition}.json").read_text())
            if not (
                summary["cohort_sha256"] == COHORT_SHA
                and summary["purpose"] == "qualitative_visualization"
            ):
                raise ValueError("Prediction summary disagrees with the fixed cohort")
            result = next(
                row for row in summary["clips"] if row["clip_id"] == clip["clip_id"]
            )
            if result["status"] != "ok":
                raise ValueError(f"Incomplete prediction {condition} {clip['clip_id']}")
            path = predictions / Path(result["npz"]).name
            if dual_sha256(path) != result["sha256"]:
                raise ValueError(f"Prediction SHA-256 mismatch: {path}")
            with np.load(path, allow_pickle=False) as raw:
                shared = {key: raw[key] for key in AXES}
                if axes is None:
                    axes = shared
                else:
                    for key in AXES:
                        np.testing.assert_array_equal(
                            axes[key], shared[key], err_msg=key
                        )
                geometry = {key: raw[key] for key in GEOMETRY}
            if (
                not np.array_equal(shared["source_frames"], clip["evaluation_frames"])
                or not np.array_equal(shared["player_ids"], clip["player_ids"])
                or not np.array_equal(shared["camera_ids"], clip["camera_ids"])
                or not np.array_equal(
                    geometry["camera_valid"], shared["segment_index"] >= 0
                )
            ):
                raise ValueError(
                    "Prediction person, camera or frame axis differs from the fixed input"
                )
            arrays.append(geometry)
            hashes[condition] = result["sha256"]
        return cls(clip, axes, tuple(arrays), hashes)


def label(frame, text, x, y, color=INK, size=0.48):
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA
    )


def boxed(image, lines):
    width = image.shape[1]
    header = np.zeros((57, width, 3), np.uint8)
    header[:] = (26, 27, 30)
    title, title_color = lines[0]
    label(header, title, 10, 19, title_color, size=0.56)
    for i, (text, color) in enumerate(lines[1:]):
        label(
            header, text, 10 + (i % 2) * (width // 2), 36 + (i // 2) * 16, color, 0.45
        )
    return np.vstack((header, image, np.zeros((1, width, 3), np.uint8)))


def projected(points, intrinsic):
    values = points @ intrinsic.T
    front = values[:, 2] > 1e-6
    xy = np.zeros((len(points), 2), np.float64)
    np.divide(values[:, :2], values[:, 2, None], out=xy, where=front[:, None])
    return xy, front


def dots(image, points, intrinsic, color):
    h, w = image.shape[:2]
    uv, front = projected(points, intrinsic)
    for x, y in uv[
        front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ]:
        cv2.circle(image, (round(x), round(y)), 2, color, -1, cv2.LINE_AA)
    return image


def missing(geometry, player, row):
    if not geometry["root_valid"][player, row]:
        return f"no placement (R{geometry['placement_root_reason'][player, row]})"
    if not geometry["heading_valid"][player, row]:
        return "no heading"
    return "mesh rejected"


def overlays(record, row, originals, renderer, mode):
    ids, selected = record.axes["player_ids"], record.axes["selected_view_index"]
    scale = np.diag([WIDTH / record.clip["width"], HEIGHT / record.clip["height"], 1])
    panels = []
    for view, frame in enumerate(originals):
        camera = str(record.axes["camera_ids"][view])
        source = int(record.axes["source_frames"][row])
        titles = [
            (
                f"P{int(identity)} {'SRC' if selected[p] == view else record.axes['camera_ids'][selected[p]]}",
                PALETTE[p],
            )
            for p, identity in enumerate(ids)
        ]
        original = boxed(frame, [(f"{camera}  original  frame {source}", INK), *titles])
        choices = [original]
        for name, mesh in zip(MODELS, record.predictions, strict=True):
            image = frame.copy()
            people, shapes = [], []
            for p, identity in enumerate(ids):
                tag = f"P{int(identity)} " + (
                    "SRC"
                    if selected[p] == view
                    else f"from {record.axes['camera_ids'][selected[p]]}"
                )
                if mode == "before":
                    if selected[p] != view:
                        people.append((tag, PALETTE[p]))
                        continue
                    if not mesh["camera_valid"][p, row]:
                        people.append((f"{tag} no estimate", PALETTE[p]))
                        continue
                    positions = mesh["camera_vertices_m"][p, row]
                    intrinsic = scale @ mesh["camera_mesh_K"][p]
                else:
                    if not mesh["court_valid"][p, row]:
                        people.append((f"{tag} {missing(mesh, p, row)}", PALETTE[p]))
                        continue
                    field = "court_coco17_m" if mode == "joints" else "court_vertices_m"
                    positions = (
                        mesh[field][p, row] @ record.axes["R"][view].T
                        + record.axes["t"][view]
                    )
                    intrinsic = scale @ record.axes["K"][view]
                shapes.append((positions, intrinsic, PALETTE[p]))
                gap = (
                    " (interpolated crop)"
                    if mode == "before"
                    and not record.axes["selected_view_observed"][p, row]
                    else ""
                )
                people.append((tag + gap, PALETTE[p]))
            for positions, intrinsic, color in sorted(
                shapes, key=lambda item: -float(item[0][:, 2].mean())
            ):
                image = (
                    dots(image, positions, intrinsic, color)
                    if mode == "joints"
                    else renderer.render_overlay(
                        image,
                        positions,
                        intrinsic,
                        color=tuple(x / 255 for x in color),
                        alpha=0.63,
                    )
                )
            choices.append(
                boxed(image, [(f"{camera}  {name}  frame {source}", INK), *people])
            )
        panels.append(np.hstack(choices))
    return np.vstack(panels)


def virtual_court(width, height):
    eye = np.array([19.0, -25.0, 24.0])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    focal = width * 0.8
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    return PinholeCamera("fixed-court", k, rotation, -rotation @ eye)


def mesh_comparison(record, row, renderer, court):
    width, height = 832, 448
    ids = record.axes["player_ids"]
    before_panels, after_panels = [], []
    for name, mesh in zip(MODELS, record.predictions, strict=True):
        pre = np.full((168, width, 3), (32, 35, 43), np.uint8)
        after = np.full((height, width, 3), (41, 47, 55), np.uint8)
        kp = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()
        xy, front = court.project(kp)
        for a, b in COURT_SKELETON:
            if front[a] and front[b]:
                cv2.line(
                    after,
                    tuple(xy[a].round().astype(int)),
                    tuple(xy[b].round().astype(int)),
                    (102, 115, 125),
                    1,
                    cv2.LINE_AA,
                )
        state, shapes = [], []
        for p, identity in enumerate(ids):
            color = tuple(value / 255 for value in PALETTE[p])
            camera = str(
                record.axes["camera_ids"][record.axes["selected_view_index"][p]]
            )
            if mesh["camera_valid"][p, row]:
                points = mesh["camera_vertices_m"][p, row]
                hip = mesh["camera_coco17_m"][p, row, [11, 12]].mean(axis=0)
                # Keep the recovered orientation in the selected camera axes.
                # Place both models' pelvis at the same depth for comparison.
                local = points - hip + [0, 0, 4.0]
                k = np.array([[255, 0, 104 + 208 * p], [0, 255, 84], [0, 0, 1]])
                pre = renderer.render_overlay(pre, local, k, color=color, alpha=0.96)
            label(
                pre,
                f"P{int(identity)}  {camera}"
                + ("" if mesh["camera_valid"][p, row] else "  no estimate"),
                10 + p * 208,
                157,
                PALETTE[p],
            )
            if mesh["court_valid"][p, row]:
                positions = (
                    mesh["court_vertices_m"][p, row] @ court.rotation.T
                    + court.translation
                )
                shapes.append((positions, color))
            state.append(
                (
                    f"P{int(identity)}  "
                    + (
                        "placed"
                        if mesh["court_valid"][p, row]
                        else missing(mesh, p, row)
                    ),
                    PALETTE[p],
                )
            )
        for positions, color in sorted(
            shapes, key=lambda item: -float(item[0][:, 2].mean())
        ):
            after = renderer.render_overlay(
                after, positions, court.intrinsic, color=color, alpha=0.96
            )
        source = int(record.axes["source_frames"][row])
        before_panels.append(
            boxed(
                pre,
                [
                    (
                        f"{name}  frame {source}  |  before - camera axes, pelvis centered",
                        INK,
                    )
                ],
            )
        )
        after_panels.append(
            boxed(
                after,
                [
                    (f"{name}  frame {source}  |  after - common court view", INK),
                    *state,
                ],
            )
        )
    return np.vstack((np.hstack(before_panels), np.hstack(after_panels)))


def stream_frame(iterator, index, size):
    packet = next(iterator)
    while packet.index < index:
        packet = next(iterator)
    if packet.index != index or packet.original_size != size:
        raise ValueError(f"Decoded frame does not match fixed source frame {index}")
    return cv2.resize(packet.frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def render_clip(clip, predictions, output, faces_path, preview=False):
    cv2.setNumThreads(1)
    record = Comparison.read(clip, predictions)
    name = str(clip["clip_id"]).split("/")[-1]
    paths = tuple(pinned_file(SOURCE, video) for video in clip["videos"])
    renderer = MeshRenderer(load_smpl_faces(faces_path.resolve(strict=True)))
    if renderer.faces.min() < 0 or renderer.faces.max() >= 6890:
        raise ValueError("Expected SMPL 6890-vertex topology")
    output.mkdir(parents=True, exist_ok=True)
    indices = record.axes["source_frames"]
    end = min(int(indices[-1]) + 2, clip["source_frames"])
    durations = np.diff(indices, append=end)
    if (durations <= 0).any():
        raise ValueError("Unexpected source frame order")
    if preview:
        indices, durations = indices[:2], durations[:2]
    fps = clip["fps"]
    court = virtual_court(832, 448)
    media = {}
    with ExitStack() as stack:
        readers = [
            stack.enter_context(closing(iter(OpenCVVideoFrameReader(path))))
            for path in paths
        ]
        files = {stage: output / f"{name}-{stage}.mp4" for stage in PRODUCTS}
        temporary = {
            key: file.with_name(file.stem + ".partial.mp4")
            for key, file in files.items()
        }
        writers = {
            key: stack.enter_context(VideoWriter(file, fps=fps, crf=21))
            for key, file in temporary.items()
        }
        for row, (index, count) in enumerate(zip(indices, durations, strict=True)):
            originals = [
                stream_frame(reader, index, (clip["width"], clip["height"]))
                for reader in readers
            ]
            for stage, writer in writers.items():
                frame = (
                    mesh_comparison(record, row, renderer, court)
                    if stage == "3d"
                    else overlays(record, row, originals, renderer, stage)
                )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for _ in range(int(count)):
                    writer.write_frame(rgb)
                if preview and row == 0:
                    assert cv2.imwrite(str(output / f"{name}-{stage}.png"), frame)
    for stage, file in files.items():
        temporary[stage].replace(file)
        media[stage] = {"file": file.name, "sha256": dual_sha256(file)}
    valid = {
        label: {
            "before": int(mesh["camera_valid"].sum()),
            "after": int(mesh["court_valid"].sum()),
        }
        for label, mesh in zip(MODELS, record.predictions, strict=True)
    }
    print(
        f"{name}: {len(indices)} evaluation frames, {int(durations.sum())} encoded source ticks, {len(media)} videos",
        flush=True,
    )
    return {
        "clip": name,
        "input_sha256": record.digests,
        "files": media,
        "evaluation_frames": indices.tolist(),
        "fps": fps,
        "encoded_frames": int(durations.sum()),
        "mesh_available": valid,
    }


def index_page(records):
    header = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FastHMRとGVHMRの同期比較</title>
<style>body{font:16px/1.65 system-ui;color:#192129;background:#f5f6f7;margin:3vw auto;max-width:1000px;padding:0 20px} table{border-collapse:collapse;background:white} th,td{padding:9px 12px;border-bottom:1px solid #dbe0e5}th{text-align:left}a{color:#135b95}code{font-size:.9em}</style>
</head><body><h1>FastHMRとGVHMRの同期比較</h1>
<p>Meiji 4camの固定6 clip。各カメラ段の左から元映像、保存済みGVHMR、FastHMR CameraHMR PT16。
<code>SRC</code> はメッシュ推定に使ったカメラです。配置前はそのビューに投影し、配置後はコート座標から4ビューに投影します。</p>
<p>3D動画の上段はカメラ相対メッシュを骨盤位置で揃えた表示、下段は共通コート視点です。選手IDの色は動画間で対応します。
元映像の時刻を保ち、推論サンプル間は直前の表示を保持します。推定と配置の欠測は各パネルに表示します。</p>
<p>映像を使った目視確認用。独立した人体メッシュの正解はありません。</p>
<table><thead><tr><th>clip</th>"""
    header += (
        "".join(f"<th>{title}</th>" for title in PRODUCTS.values())
        + "</tr></thead><tbody>"
    )
    rows = []
    for item in records:
        links = "".join(
            f"<td><a href='{escape(x['file'], quote=True)}'>再生</a></td>"
            for x in item["files"].values()
        )
        rows.append(f"<tr><th>{escape(item['clip'])}</th>{links}</tr>")
    return header + "".join(rows) + "</tbody></table></body></html>\n"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    records = [record for record, _ in cohort_records()]
    if args.preview:
        result = [
            render_clip(records[0], args.predictions, args.output, args.faces, True)
        ]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            work = [
                pool.submit(render_clip, row, args.predictions, args.output, args.faces)
                for row in records
            ]
            result = [item.result() for item in work]
    (args.output / "index.html").write_text(index_page(result), encoding="utf-8")
    write_json_atomic(
        args.output / "manifest.json",
        {
            "purpose": "qualitative_visualization",
            "cohort_sha256": COHORT_SHA,
            "faces_sha256": dual_sha256(args.faces),
            "preview": args.preview,
            "clips": result,
        },
    )


if __name__ == "__main__":
    main()
