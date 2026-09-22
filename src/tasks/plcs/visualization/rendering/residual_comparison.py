"""Render PLCS residual predictions against their triangulated initialization."""

from __future__ import annotations

import html
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from src.tasks.plcs.visualization.adapters.residual import (
    ResidualComparison,
    load_residual_comparison,
)
from src.utils.rendering.court_renderer import CourtRenderer, CourtStyle
from src.utils.rendering.skeleton_renderer import SkeletonRenderer, SkeletonStyle
from src.utils.schema.player import COCO17_SKELETON
from src.utils.video import RandomAccessVideoReader, VideoWriter

_BACKGROUND = "#101924"
_FOREGROUND = "#e7edf5"
_RAW = "#97a6b8"
_COLORS = ("#29d4ba", "#ffaf61")
_MIN_SCORE = 0.3
_WIDTH, _HEIGHT = 1440, 960


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _mean(values: np.ndarray, valid: np.ndarray, axis: Any) -> np.ndarray:
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0).sum(axis=axis)
    result: np.ndarray = np.divide(
        total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0
    )
    return result


def _style_axes(axis: Any, *, three_dimensional: bool = False) -> None:
    axis.set_facecolor(_BACKGROUND)
    axis.tick_params(colors=_FOREGROUND, labelsize=8)
    axis.xaxis.label.set_color(_FOREGROUND)
    axis.yaxis.label.set_color(_FOREGROUND)
    axis.title.set_color(_FOREGROUND)
    if three_dimensional:
        axis.zaxis.label.set_color(_FOREGROUND)
        for dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
            dimension.set_pane_color((0.08, 0.13, 0.19, 1))
    else:
        for spine in axis.spines.values():
            spine.set_color("#41516a")
    axis.grid(True, alpha=0.2)


def _reprojection(
    data: ResidualComparison,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = (
        (data.scores >= _MIN_SCORE)
        & np.isfinite(data.observations).all(-1)
        & data.initial_projected_valid
        & data.prediction_projected_valid
    )
    before = np.linalg.norm(data.initial_px - data.observations, axis=-1)
    after = np.linalg.norm(data.prediction_px - data.observations, axis=-1)
    return before, after, common


def _save_diagnostics(data: ResidualComparison, path: Path) -> tuple[float, float]:
    figure = Figure(figsize=(14.4, 9), dpi=120, facecolor=_BACKGROUND)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.9, hspace=0.43)
    figure.suptitle(
        f"{data.task} | observed-image diagnostics (no independent 3D GT)",
        color=_FOREGROUND,
    )
    time = np.arange(data.frames) / data.fps
    before, after, common = _reprojection(data)
    axes[0].plot(
        time, _mean(before, common, (0, 1, 3)), color=_RAW, label="Raw triangulation"
    )
    axes[0].plot(
        time,
        _mean(after, common, (0, 1, 3)),
        color=_COLORS[0],
        label="Residual prediction",
    )
    axes[0].set_title(
        "Mean reprojection | same raw-valid points, observation score >= 0.3",
        fontsize=11,
    )
    axes[0].set_ylabel("Pixels")
    correction = np.linalg.norm(data.prediction - data.initial, axis=-1)
    for person in range(len(data.initial)):
        axes[1].plot(
            time,
            _mean(correction[person], data.initial_valid[person], -1),
            color=_COLORS[person],
            label=f"P{person}",
        )
        axes[2].plot(
            time,
            100 * data.initial_valid[person].mean(-1),
            color=_COLORS[person],
            label=f"P{person}",
        )
    axes[1].set_title("Mean correction magnitude | raw-valid points only", fontsize=11)
    axes[1].set_ylabel("Metres")
    axes[2].set_title(
        "Valid raw triangulation | filled seeds count as missing", fontsize=11
    )
    axes[2].set_ylabel("Percent of points")
    axes[2].set_ylim(-2, 102)
    axes[2].set_xlabel("Clip time (s)")
    for axis in axes:
        _style_axes(axis)
        axis.legend(facecolor=_BACKGROUND, labelcolor=_FOREGROUND, fontsize=9)
    figure.savefig(path, facecolor=_BACKGROUND)
    figure.clear()
    return float(_mean(before, common, None)), float(_mean(after, common, None))


def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _camera_image(
    data: ResidualComparison, frame: int, view: int, image: np.ndarray
) -> np.ndarray:
    panel_width = _WIDTH // len(data.camera_ids)
    panel_height = round(panel_width * data.height / data.width)
    canvas = cv2.resize(
        image[..., ::-1], (panel_width, panel_height), interpolation=cv2.INTER_AREA
    )
    scale = np.asarray([panel_width / data.width, panel_height / data.height])

    def draw(
        points: np.ndarray,
        valid: np.ndarray,
        color: str,
        initial_missing: np.ndarray | None = None,
    ) -> None:
        inside = (
            valid & (points >= 0).all(-1) & (points < [data.width, data.height]).all(-1)
        )
        pixels = np.rint(np.where(inside[:, None], points, 0) * scale).astype(np.int32)
        for start, end in COCO17_SKELETON:
            if inside[start] and inside[end]:
                cv2.line(
                    canvas,
                    tuple(pixels[start]),
                    tuple(pixels[end]),
                    _rgb(color),
                    2,
                    cv2.LINE_AA,
                )
        for joint in np.flatnonzero(inside):
            point = tuple(pixels[joint])
            if initial_missing is not None and initial_missing[joint]:
                cv2.drawMarker(
                    canvas,
                    point,
                    _rgb(color),
                    cv2.MARKER_TILTED_CROSS,
                    8,
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    canvas,
                    point,
                    3,
                    _rgb(color),
                    -1,
                    cv2.LINE_AA,
                )

    for person in range(len(data.initial)):
        draw(
            data.initial_px[person, view, frame],
            data.initial_projected_valid[person, view, frame],
            _RAW,
        )
        draw(
            data.prediction_px[person, view, frame],
            data.prediction_projected_valid[person, view, frame],
            _COLORS[person],
            ~data.initial_valid[person, frame],
        )
        observed = data.observations[person, view, frame]
        available = (data.scores[person, view, frame] > 0) & np.isfinite(observed).all(
            -1
        )
        available &= (observed >= 0).all(-1) & (
            observed < [data.width, data.height]
        ).all(-1)
        for joint in np.flatnonzero(available):
            center = tuple(np.rint(observed[joint] * scale).astype(np.int32))
            reliable = data.scores[person, view, frame, joint] >= _MIN_SCORE
            cv2.circle(
                canvas,
                center,
                3 if reliable else 2,
                (255, 255, 255) if reliable else (160, 160, 160),
                1,
                cv2.LINE_AA,
            )
    result: np.ndarray = np.asarray(canvas, dtype=np.uint8)
    return result


class _ComparisonCanvas:
    def __init__(self, data: ResidualComparison) -> None:
        self.data = data
        self.figure = Figure(
            figsize=(_WIDTH / 100, _HEIGHT / 100), dpi=100, facecolor=_BACKGROUND
        )
        self.canvas = FigureCanvasAgg(self.figure)
        self.figure.text(
            0.025,
            0.966,
            f"{data.task}  |  Raw triangulation + residual prediction",
            color=_FOREGROUND,
            fontsize=21,
            weight="bold",
        )
        for x, text, color in (
            (0.025, "o  Original 2D", "white"),
            (0.22, "Raw triangulation", _RAW),
            (0.42, "P0 prediction", _COLORS[0]),
            (0.60, "P1 prediction" if len(data.initial) > 1 else "", _COLORS[1]),
        ):
            self.figure.text(x, 0.922, text, color=color, fontsize=11)
        self.figure.text(
            0.025,
            0.022,
            "Neural residual prediction; no independent 3D ground truth.  x marks missing raw triangulation.",
            color=_RAW,
            fontsize=10,
        )
        self.status = self.figure.text(0.025, 0.531, "", color=_FOREGROUND, fontsize=11)
        self.images = []
        for view, camera_id in enumerate(data.camera_ids):
            axis: Any = self.figure.add_axes(
                (
                    view / len(data.camera_ids) + 0.008,
                    0.578,
                    1 / len(data.camera_ids) - 0.016,
                    0.303,
                )
            )
            self.images.append(
                axis.imshow(np.zeros((data.height, data.width, 3), dtype=np.uint8))
            )
            axis.set_title(camera_id, color=_FOREGROUND, fontsize=11)
            axis.axis("off")
        self.world_axis = cast(
            Any,
            self.figure.add_axes((0.005, 0.083, 0.49, 0.40), projection="3d"),
        )
        CourtRenderer(
            CourtStyle(
                line_color="#7f91a5",
                line_width=1,
                court_color="#16372f",
                surface_alpha=0.3,
            )
        ).render_3d(self.world_axis, show_net=False, show_apron=False, set_limits=False)
        self.world_axis.set(
            xlim=(-12, 12),
            ylim=(-22, 22),
            zlim=(-1, 12),
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
            title="Physical court coordinates",
        )
        self.world_axis.set_box_aspect((24, 44, 13))
        self.world_axis.view_init(elev=25, azim=-62)
        _style_axes(self.world_axis, three_dimensional=True)
        self.detail_axes: list[Any] = []
        self.cursors: list[Any] = []
        for person in range(len(data.initial)):
            axis = self.figure.add_axes(
                (
                    0.47 + person * 0.47 / len(data.initial),
                    0.085,
                    0.44 / len(data.initial),
                    0.40,
                ),
                projection="3d",
            )
            axis.set(
                xlim=(-1.6, 1.6),
                ylim=(-1.6, 1.6),
                zlim=(-1.6, 1.6),
                xlabel="X offset (m)",
                ylabel="Y offset (m)",
                zlabel="Z offset (m)",
            )
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=15, azim=-65)
            _style_axes(axis, three_dimensional=True)
            self.detail_axes.append(axis)
        self.dynamic_axes = [self.world_axis, *self.detail_axes]
        self.static_counts = [
            (len(axis.lines), len(axis.collections)) for axis in self.dynamic_axes
        ]
        self.skeleton = SkeletonRenderer("coco17")

    def _pose(
        self,
        axis: Any,
        points: np.ndarray,
        valid: np.ndarray,
        color: str,
        *,
        missing: np.ndarray | None = None,
    ) -> None:
        self.skeleton.render_3d(
            axis,
            points,
            valid,
            style_override=SkeletonStyle(
                joint_color=color,
                bone_color=color,
                joint_size=9,
                bone_width=1.5,
                bone_alpha=0.95,
            ),
        )
        if missing is not None and (valid & missing).any():
            axis.scatter(
                *points[valid & missing].T,
                color=color,
                marker="x",
                s=36,
                linewidths=1.5,
            )

    def frame(self, index: int, images: list[np.ndarray]) -> NDArray[np.uint8]:
        data = self.data
        for view, (artist, frame) in enumerate(zip(self.images, images, strict=True)):
            if frame.shape != (data.height, data.width, 3):
                raise ValueError("Decoded frame dimensions changed")
            artist.set_data(_camera_image(data, index, view, frame))
        for axis, (lines, collections) in zip(
            self.dynamic_axes, self.static_counts, strict=True
        ):
            for artist in [
                *list(axis.lines)[lines:],
                *list(axis.collections)[collections:],
            ]:
                artist.remove()
        outside = 0
        for person in range(len(data.initial)):
            raw = data.initial[person, index]
            prediction = data.prediction[person, index]
            raw_valid = data.initial_valid[person, index]
            world_raw = (
                raw_valid
                & (raw >= [-12, -22, -1]).all(-1)
                & (raw <= [12, 22, 12]).all(-1)
            )
            world_prediction = (prediction >= [-12, -22, -1]).all(-1) & (
                prediction <= [12, 22, 12]
            ).all(-1)
            outside += int((raw_valid & ~world_raw).sum() + (~world_prediction).sum())
            self._pose(self.world_axis, raw, world_raw, _RAW)
            self._pose(
                self.world_axis,
                prediction,
                world_prediction,
                _COLORS[person],
                missing=~raw_valid,
            )
            center = prediction[[11, 12]].mean(0)
            local_raw, local_prediction = raw - center, prediction - center
            visible_raw = raw_valid & (np.abs(local_raw) <= 1.6).all(-1)
            visible_prediction = (np.abs(local_prediction) <= 1.6).all(-1)
            axis = self.detail_axes[person]
            cropped = int(
                (raw_valid & ~visible_raw).sum() + (~visible_prediction).sum()
            )
            axis.set_title(
                f"P{person} | origin: predicted hips\nDetail viewport: {cropped} points outside",
                color=_FOREGROUND,
                fontsize=9,
            )
            self._pose(axis, local_raw, visible_raw, _RAW)
            self._pose(
                axis,
                local_prediction,
                visible_prediction,
                _COLORS[person],
                missing=~raw_valid,
            )
        for cursor in self.cursors:
            cursor.set_xdata([index / data.fps] * 2)
        valid_count = int(data.initial_valid[:, index].sum())
        self.status.set_text(
            f"{data.clip_id}   |   {index / data.fps:.2f} s   |   Frame {index}/{data.frames - 1}   |   Raw valid {valid_count}/{data.initial_valid[:, index].size}   |   Court viewport: {outside} points outside"
        )
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def _write_html(
    data: ResidualComparison,
    path: Path,
    *,
    before: float,
    after: float,
    stride: int,
) -> None:
    before_text = f"{before:.2f} px" if np.isfinite(before) else "対象点なし"
    after_text = f"{after:.2f} px" if np.isfinite(after) else "対象点なし"
    title = html.escape(f"{data.task} · {data.clip_id}")
    path.write_text(
        f"""<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · 三角測量と残差補正</title>
<style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif;background:#0b111a;color:#e7edf5}}
body{{max-width:1440px;margin:auto;padding:32px 24px}}h1{{font-size:30px;margin:0 0 12px}}
p{{line-height:1.7;color:#b6c4d6}}.card{{background:#111c2b;border:1px solid #2b3b50;border-radius:14px;padding:20px;margin:22px 0}}
.legend,.links{{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0}}a{{color:#63decf}}video,img{{display:block;width:100%;border-radius:8px}}
.raw{{color:#97a6b8}}.p0{{color:#29d4ba}}.p1{{color:#ffaf61}}.metric{{font-size:24px;margin:8px 0}}small{{color:#99abc0}}
</style><main><h1>{title}</h1><p>実画像の2D観測、三角測量の初期値、モデルによる残差補正後の位置を比較します。<strong>独立した3D正解はありません。</strong></p>
<div class="legend"><span>○ 元の2D観測</span><span class="raw">灰: 三角測量</span><span class="p0">緑: P0補正後</span>{'<span class="p1">橙: P1補正後</span>' if len(data.initial) > 1 else ""}<span>×: 有効な三角測量初期値がない予測点</span></div>
<div class="card"><video controls playsinline preload="metadata" poster="comparison.png"><source src="comparison.mp4" type="video/mp4"></video>
<div class="links"><a href="comparison.mp4">比較動画</a><a href="comparison.png">代表フレーム</a><a href="diagnostics.png">診断画像</a></div>
<small>{data.frames}フレーム / {data.fps:.5f} fps。動画は{stride}フレームごとに描画し、元の時間経過で再生します。</small></div>
<div class="card"><strong>元の2D観測への平均再投影誤差</strong><div class="metric">{before_text} → {after_text}</div>
<p>観測score ≥ 0.3かつ三角測量・補正後投影がともに有効な同一の点集合で比較しています。これは画像との整合性であり、3Dの正解誤差ではありません。</p><img src="diagnostics.png" alt="再投影誤差、補正量、三角測量の有効率"></div>
<p>欠測を埋めた初期seedは灰色の三角測量として表示しません。補正後はモデルの残差予測です。人物拡大図では補正後のhip中点を共通原点としています。表示範囲外の点数は図中に明示しています。低scoreの2D観測は小さい暗色の円です。</p></main></html>""",
        encoding="utf-8",
    )


def render_comparison(
    clip_dir: Path, predictions_path: Path, output_dir: Path, stride: int = 4
) -> dict[str, str]:
    """Render an H.264 comparison, representative PNG, diagnostics and HTML.

    Inputs remain in original pixels and physical court metres. Finite filled
    seeds with ``initial_valid=False`` are excluded from every raw visualization
    and metric. Diagnostics use the same raw-valid, score>=0.3 point set before
    and after correction; they do not measure independent 3-D accuracy.
    """
    stride = _positive_int(stride, "stride")
    data = load_residual_comparison(clip_dir, predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "video": output_dir / "comparison.mp4",
        "comparison": output_dir / "comparison.png",
        "diagnostics": output_dir / "diagnostics.png",
        "html": output_dir / "index.html",
    }
    before, after = _save_diagnostics(data, paths["diagnostics"])
    frame_indices = list(range(0, data.frames, stride))
    representative = frame_indices[len(frame_indices) // 2]
    canvas = _ComparisonCanvas(data)
    try:
        with ExitStack() as stack:
            readers = [
                stack.enter_context(RandomAccessVideoReader(path))
                for path in data.videos
            ]
            writer = stack.enter_context(
                VideoWriter(paths["video"], fps=data.fps / stride, crf=19)
            )
            for index in frame_indices:
                frame = canvas.frame(index, [reader.read(index) for reader in readers])
                writer.write_frame(frame)
                if index == representative and not cv2.imwrite(
                    str(paths["comparison"]), frame[..., ::-1]
                ):
                    raise RuntimeError(
                        "Failed to write representative comparison image"
                    )
    finally:
        canvas.figure.clear()
    _write_html(data, paths["html"], before=before, after=after, stride=stride)
    return {name: str(path.resolve()) for name, path in paths.items()}
