"""Apply three temporal ball smoothers to a committed clip and export mesh-ready scenes.

The source scene and component artifacts remain immutable. Each derived scene retains
the original player mesh, masks, and court, replacing only the ball's 3D positions.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.pipeline.components.ball_smoothing import (
    BallSmoothingConfig,
    BallSmoothingInput,
    BallSmoothingMethod,
    BallSmoothingModule,
    ball_event_frames,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationOutput
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256

METHODS: tuple[BallSmoothingMethod, ...] = ("savgol", "robust_spline", "ballistic_rts")
LABELS = {"savgol": "局所2次多項式", "robust_spline": "ロバスト加速度正則化", "ballistic_rts": "重力付きRTS平滑化"}
COLORS = {"raw": "#9ca3af", "savgol": "#2196f3", "robust_spline": "#ff9800", "ballistic_rts": "#4caf50"}


def _load_component(root: Path, index: dict[str, Any], name: str, output_type: type[Any]) -> Any:
    reference = index["artifacts"][name]
    path = root / reference["path"]
    if dual_sha256(path) != reference["sha256"]:
        raise ValueError(f"Component descriptor checksum mismatch: {name}")
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    if descriptor["artifact_id"] != reference["artifact_id"]:
        raise ValueError(f"Component artifact ID mismatch: {name}")
    return ArtifactCodec(output_type).load(descriptor["payload"], path.parent, descriptor["arrays"])


def _source(index: dict[str, Any]) -> ClipSource:
    data = index["source"]
    videos = tuple(SourceVideo(camera_id=item["camera_id"], path=Path(item["path"]), sha256=item["sha256"],
        num_frames=item["num_frames"], fps=item["fps"], width=item["width"], height=item["height"])
        for item in data["videos"])
    return ClipSource(data["clip_id"], videos)


def _quality(
    raw: NDArray[np.float32], current: NDArray[np.float32], valid: NDArray[np.bool_],
    inliers: NDArray[np.bool_], reprojection_px: NDArray[np.float32], fps: float,
    events: NDArray[np.int32],
) -> dict[str, float | int]:
    triplets = valid[:-2] & valid[1:-1] & valid[2:]
    flight = triplets.copy()
    for event in events:
        flight[max(0, int(event) - 2):min(len(flight), int(event) + 1)] = False
    acceleration = np.linalg.norm(np.diff(current, n=2, axis=0) * fps * fps, axis=-1)
    displacement = np.linalg.norm(current[valid] - raw[valid], axis=-1)
    projected = reprojection_px[inliers & valid[None, :]]
    if not len(projected) or not flight.any():
        raise ValueError("Clip lacks supported flight and multiview reprojection samples")
    return {
        "valid_frames": int(valid.sum()),
        "event_frames": int(len(events)),
        "flight_acceleration_p50_mps2": float(np.median(acceleration[flight])),
        "flight_acceleration_p95_mps2": float(np.quantile(acceleration[flight], .95)),
        "displacement_p50_m": float(np.median(displacement)),
        "displacement_p95_m": float(np.quantile(displacement, .95)),
        "displacement_max_m": float(displacement.max()),
        "reprojection_p50_px": float(np.median(projected)),
        "reprojection_p95_px": float(np.quantile(projected, .95)),
        "reprojection_over_20px": int((projected > 20).sum()),
        "reprojection_samples": int(len(projected)),
    }


def _diagnostic_plot(output_dir: Path, trajectories: dict[str, NDArray[np.float32]], valid: NDArray[np.bool_],
                     fps: float, events: NDArray[np.int32]) -> None:
    time = np.arange(len(valid)) / fps
    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
    for axis, label in zip(axes[:3], ("X (m)", "Y (m)", "Z (m)"), strict=True):
        coordinate = ("X (m)", "Y (m)", "Z (m)").index(label)
        for method, points in trajectories.items():
            axis.plot(time, np.where(valid, points[:, coordinate], np.nan),
                color=COLORS[method], linewidth=.7 if method == "raw" else 1.2,
                alpha=.65 if method == "raw" else .95, label=method)
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    for method, points in trajectories.items():
        speed = np.linalg.norm(np.diff(points, axis=0) * fps, axis=-1)
        speed[~(valid[1:] & valid[:-1])] = np.nan
        axes[3].plot(time[1:], speed, color=COLORS[method], linewidth=.7 if method == "raw" else 1.1,
            alpha=.65 if method == "raw" else .95)
    axes[3].set_ylabel("Speed (m/s)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(alpha=.2)
    for axis in axes:
        for event in events:
            axis.axvline(float(event) / fps, color="#e11d48", alpha=.13, linewidth=.8)
    axes[0].legend(ncol=4)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_comparison.png", dpi=140)
    plt.close(fig)


def _write_html(output_dir: Path, report: dict[str, Any]) -> None:
    rows = []
    for method in METHODS:
        metrics = report["variants"][method]["quality"]
        rows.append(f"<tr><th>{html.escape(LABELS[method])}</th><td>{metrics['flight_acceleration_p95_mps2']:.1f}</td>"
                    f"<td>{metrics['displacement_p95_m']:.3f}</td><td>{metrics['reprojection_p95_px']:.2f}</td>"
                    f"<td>{metrics['reprojection_over_20px']}</td></tr>")
    cards = []
    for method in METHODS:
        name = html.escape(LABELS[method])
        cards.append(f"<article><h2>{name}</h2><p>SMPL mesh 2名・球軌道・コートを同じ視点で表示</p>"
                     f"<video controls preload='metadata' src='{method}/mesh_full.mp4'></video>"
                     f"<a href='{method}/mesh_preview.png'><img src='{method}/mesh_preview.png' alt='{name} mesh preview'></a>"
                     f"<p><a href='{method}/scene.npz'>scene.npz</a> · <a href='{method}/trajectory.npz'>trajectory.npz</a></p></article>")
    content = """<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>球三角測量の時系列平滑化 3方式</title>
<style>body{font:16px system-ui,sans-serif;background:#0c1520;color:#e5edf6;margin:2rem auto;max-width:1400px;padding:0 1rem}
h1,h2{color:#fff}p{line-height:1.6}a{color:#7dd3fc}table{border-collapse:collapse;width:100%;margin:1.5rem 0}
td,th{border-bottom:1px solid #365064;padding:.65rem;text-align:right}th{text-align:left}article{background:#172638;border-radius:12px;padding:1.25rem;margin:1.5rem 0}
video,img{display:block;width:min(100%,1000px);margin:.8rem auto;border-radius:8px}small{color:#9db0c2}</style>
<h1>球三角測量の時系列平滑化 3方式</h1>
<p>同じ実クリップの確定済み2名、同じコート・mesh・カメラで球の3D軌道だけを置き換えた比較です。欠測は補間せず、バウンドと打球方向の反転点を保持します。</p>
<table><thead><tr><th>方式</th><th>飛行中加速度 p95 (m/s²)</th><th>元の3Dからの移動 p95 (m)</th><th>再投影誤差 p95 (px)</th><th>20px超の観測数</th></tr></thead><tbody>
""" + "".join(rows) + "</tbody></table><p><a href='trajectory_comparison.png'><img src='trajectory_comparison.png' alt='3D ball trajectories and speed comparison'></a></p>" + "".join(cards) + """
<small>正解3Dは未提供です。加速度・再投影は平滑さと観測整合を測る診断値であり、3D精度の保証ではありません。元の三角測量artifactとscene exportは変更していません。</small></html>"""
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def compare(scene_index: Path, output_dir: Path) -> dict[str, Any]:
    if scene_index.name != "scene.json":
        raise ValueError("--scene-index must point to a committed scene.json")
    index = json.loads(scene_index.read_text(encoding="utf-8"))
    root = scene_index.parent
    scene = load_scene_result(scene_index)
    triangulation = _load_component(root, index, "ball_triangulation", BallTriangulationOutput)
    alignment = _load_component(root, index, "camera_alignment", CameraAlignmentOutput)
    source = _source(index)
    raw = triangulation.ball
    if raw is None or alignment.geometry is None or scene.ball_3d is None or scene.ball_3d_valid is None:
        raise ValueError("Scene must contain aligned multiview ball triangulation")
    if scene.smpl_vertices_local is None or scene.player_smpl_valid is None or not scene.player_smpl_valid.any():
        raise ValueError("Scene must contain player SMPL meshes for mesh visualization")
    if not np.array_equal(scene.ball_3d, raw.trajectory.positions) or not np.array_equal(scene.ball_3d_valid, raw.trajectory.valid):
        raise ValueError("Scene ball and selected raw triangulation artifact disagree")
    if source.num_frames != scene.num_frames or abs(source.fps - scene.fps) > 1e-6:
        raise ValueError("Scene and component source timelines disagree")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = BallSmoothingConfig()
    events = ball_event_frames(raw.trajectory.positions, raw.trajectory.valid, config)
    report: dict[str, Any] = {
        "source_clip": source.clip_id,
        "source_scene_index": str(scene_index.resolve()),
        "raw_ball_artifact_id": index["artifacts"]["ball_triangulation"]["artifact_id"],
        "camera_alignment_artifact_id": index["artifacts"]["camera_alignment"]["artifact_id"],
        "fps": scene.fps,
        "frames": scene.num_frames,
        "player_mesh_valid_frames": scene.player_smpl_valid.sum(axis=1).astype(int).tolist(),
        "event_indices": events.tolist(),
        "raw_quality": _quality(raw.trajectory.positions, raw.trajectory.positions, raw.trajectory.valid,
            raw.trajectory.inliers, raw.trajectory.reprojection_px, scene.fps, events),
        "variants": {},
    }
    trajectories = {"raw": raw.trajectory.positions}
    for method in METHODS:
        settings = replace(config, method=method)
        output = BallSmoothingModule(settings).process(BallSmoothingInput(source, triangulation, alignment))
        ball = output.ball
        if ball is None:
            raise ValueError(f"Ball smoothing unexpectedly returned no trajectory: {method}")
        result = ball.trajectory
        if not np.array_equal(result.valid, raw.trajectory.valid) or not np.array_equal(result.positions[events], raw.trajectory.positions[events]):
            raise ValueError(f"Smoother changed support or protected events: {method}")
        variant_dir = output_dir / method
        variant_dir.mkdir(exist_ok=True)
        metadata = {**scene.metadata, "ball_smoothing": {"method": method, "config": asdict(settings),
            "source_ball_artifact_id": report["raw_ball_artifact_id"], "source_scene_index": str(scene_index.resolve()),
            "event_indices": events.tolist(), "validation": "visual_comparison_without_3d_ground_truth"}}
        derived = replace(scene, ball_3d=result.positions, metadata=metadata)
        save_scene_result(derived, variant_dir / "scene.npz")
        np.savez_compressed(variant_dir / "trajectory.npz", positions=result.positions, valid=result.valid,
            reasons=result.reasons, inliers=result.inliers, reprojection_px=result.reprojection_px, event_indices=events)
        quality = _quality(raw.trajectory.positions, result.positions, result.valid, result.inliers,
            result.reprojection_px, scene.fps, events)
        report["variants"][method] = {"config": asdict(settings), "quality": quality,
            "scene": f"{method}/scene.npz", "trajectory": f"{method}/trajectory.npz",
            "mesh_video": f"{method}/mesh_full.mp4", "mesh_preview": f"{method}/mesh_preview.png"}
        trajectories[method] = result.positions
    _diagnostic_plot(output_dir, trajectories, raw.trajectory.valid, scene.fps, events)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(output_dir, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    report = compare(arguments.scene_index, arguments.output_dir)
    print(json.dumps({"output": str(arguments.output_dir), "raw": report["raw_quality"],
        "variants": {key: value["quality"] for key, value in report["variants"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
