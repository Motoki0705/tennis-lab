"""Update the offline comparison with placed SMPL bodies from the verified exports."""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import av
import cv2
import numpy as np

from src.tennis_scene.multiview_scene.inputs import INPUT_NODE, RECIPE
from src.tennis_scene.multiview_scene.report import scene_fields
from src.tennis_scene.visualization.pose_comparison.__main__ import (
    BUNDLE_LIMIT,
    VIDEO_LIMIT,
)
from src.tennis_scene.visualization.pose_comparison.inputs import (
    ClipPlan,
    ComparisonInputs,
    load_comparison,
    load_tracking,
    select_clips,
)
from src.tennis_scene.visualization.pose_comparison.rendering import (
    GUTTER,
    HEADER_H,
    SIDE_W,
    VIDEO_H,
    VIDEO_W,
    VIEW_H,
    VIEW_W,
    FrameRenderer,
    _label,
)
from src.tennis_scene.visualization.pose_comparison.report import (
    tracked_clip_metrics,
    write_page,
)
from src.tennis_scene.visualization.rendering.multiview import sampled_frame_indices
from src.tennis_scene.visualization.rendering.smpl_court import (
    load_smpl_joint_regressor,
    smpl_vertices_to_court,
)
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.video.reader import RandomAccessVideoReader
from src.utils.video.writer import VideoWriter

SMPL_FIELDS = (
    "smpl_vertices_local",
    "smpl_global_orient",
    "player_position",
    "player_yaw",
)
ROOT_ASSET = "smpl_smpl_neutral_joint_regressor"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_bodies(store: StoreReader, digest: str) -> dict[str, Any]:
    fields = scene_fields(store, ("player_smpl_valid",))
    if fields["player_smpl_valid"].any():
        settings = store.descriptor("body_placement")["identity"]["settings"]
        assets = settings.get("assets") or settings["models"]["body"]["assets"]
        if assets[ROOT_ASSET]["sha256"] != digest:
            raise ValueError(
                f"Body regressor differs from placement: {store.source['clip_id']}"
            )
        fields.update(scene_fields(store, SMPL_FIELDS))
    return fields


def draw_bodies(
    canvas: np.ndarray,
    frames: tuple[np.ndarray, ...],
    data: ComparisonInputs,
    bodies: dict[str, Any],
    frame: int,
    origin_x: int,
    root_regressor: np.ndarray,
    kernel: np.ndarray,
) -> None:
    players = np.flatnonzero(bodies["player_smpl_valid"][:, frame])
    _label(canvas, f"SMPL {len(players)}", (origin_x + SIDE_W - 95, 20), scale=0.52)
    if players.size == 0:
        return
    vertices = smpl_vertices_to_court(
        *(bodies[name][players, frame : frame + 1] for name in SMPL_FIELDS),
        root_regressor,
    )[:, 0]
    if not np.isfinite(vertices).all():
        raise ValueError(f"Non-finite valid body: {data.source.clip_id} frame {frame}")
    for index, (camera, image) in enumerate(zip(data.cameras, frames, strict=True)):
        h, w = image.shape[:2]
        factor = min(VIEW_W / w, VIEW_H / h)
        width, height = max(1, round(w * factor)), max(1, round(h * factor))
        margin = np.array([(VIEW_W - width) // 2, (VIEW_H - height) // 2])
        uv, front = camera.project(vertices)
        pixels = np.rint(uv * (width / w, height / h) + margin)
        visible = front & np.all(
            (pixels >= margin) & (pixels < margin + (width, height)), axis=-1
        )
        row, col = divmod(index, 2)
        y, x = HEADER_H + row * VIEW_H, origin_x + col * VIEW_W
        tile = canvas[y : y + VIEW_H, x : x + VIEW_W]
        for person_uv, mask in zip(pixels, visible, strict=True):
            xy = person_uv[mask].astype(np.intp)
            occupancy = np.zeros((VIEW_H, VIEW_W), dtype=np.uint8)
            occupancy[xy[:, 1], xy[:, 0]] = 255
            closed = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(tile, contours, -1, (232, 232, 232), 1, cv2.LINE_AA)


def render_clip(
    data: ComparisonInputs,
    plan: ClipPlan,
    tracks: Any,
    bodies: tuple[dict[str, Any], dict[str, Any]],
    entry: dict[str, Any],
    output: Path,
    step: int,
    crf: int,
    regressor: np.ndarray,
    *,
    frame_limit: int | None = None,
) -> None:
    video = output / entry["video"]
    poster = output / entry["poster"]
    frames = sampled_frame_indices(
        data.source.num_frames, start=0, stop=data.source.num_frames, step=step
    )
    if (
        len(frames) != entry["video_frames"]
        or data.source.fps / step != entry["video_fps"]
    ):
        raise ValueError(f"Comparison timeline changed: {plan.clip_id}")
    if frame_limit is not None:
        frames = frames[:frame_limit]
    poster_frame = (
        frames[0] if frame_limit is not None else entry["poster_source_frame"]
    )
    entry["poster_source_frame"] = poster_frame
    renderer = FrameRenderer(data, plan, tracks)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    with ExitStack() as stack:
        readers = [
            stack.enter_context(RandomAccessVideoReader(path))
            for path in data.source.paths
        ]
        writer = stack.enter_context(
            VideoWriter(video, fps=entry["video_fps"], crf=crf)
        )
        for frame in frames:
            images = tuple(reader.read(frame) for reader in readers)
            canvas = renderer.render(frame, images)
            for x, body in zip((0, SIDE_W + GUTTER), bodies, strict=True):
                draw_bodies(canvas, images, data, body, frame, x, regressor, kernel)
            writer.write_frame(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            if frame == poster_frame and not cv2.imwrite(
                str(poster), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88]
            ):
                raise OSError(f"Could not write {poster}")
    if not poster.is_file():
        raise ValueError(f"Missing poster: {plan.clip_id}")
    entry["video_bytes"] = video.stat().st_size
    entry["video_sha256"] = dual_sha256(video)
    entry["video_frames"] = len(frames)


def check_video(path: Path, entry: dict[str, Any]) -> None:
    if path.stat().st_size != entry["video_bytes"] or path.stat().st_size > VIDEO_LIMIT:
        raise ValueError(f"Video-size check failed: {path}")
    with av.open(str(path)) as container:
        (stream,) = container.streams.video
        if (
            stream.codec_context.name != "h264"
            or (stream.width, stream.height, stream.frames)
            != (VIDEO_W, VIDEO_H, entry["video_frames"])
            or not np.isclose(
                float(stream.average_rate), entry["video_fps"], rtol=0, atol=1e-6
            )
        ):
            raise ValueError(f"Invalid comparison stream: {path}")


def build(args: argparse.Namespace) -> None:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    page = read_json(args.page / "manifest.json")
    (progress_path,) = (args.i2363_run / "all/progress").glob("*/progress.json")
    progress = read_json(progress_path)
    source, updated = page["stages"]
    pose_record = read_json(args.pose_record)
    track_record = read_json(args.track_record)
    plans, _ = select_clips(progress, page["selection"]["seed"])
    if (
        [plan.clip_id for plan in plans] != [clip["id"] for clip in source["clips"]]
        or [plan.clip_id for plan in plans] != [clip["id"] for clip in updated["clips"]]
        or dual_sha256(progress_path) != page["legacy_progress"]["sha256"]
        or dual_sha256(args.pose_record) != source["pose_run"]["sha256"]
    ):
        raise ValueError("Comparison cohort or candidate record changed")
    step, crf = (updated["encoding"][key] for key in ("frame_step", "crf"))
    root_digest = dual_sha256(args.regressor)
    regressor = load_smpl_joint_regressor(args.regressor)[:1]
    cv2.setNumThreads(1)
    updated["label"] = "段階2 · 3D追跡・選手選別・GVHMR身体"
    updated["description"] = (
        "新経路の色は時間IDです。Pは選手、Tは選外です。"
        "白い輪郭は配置が有効なGVHMR身体で、SMPLの数字はその時点の有効人数です。"
    )
    updated["tracking_run"] = {
        "name": args.track_record.name,
        "sha256": dual_sha256(args.track_record),
    }
    updated["body_asset"] = {"name": args.regressor.name, "sha256": root_digest}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".pose-comparison-bodies-", dir=args.output.parent
    ) as path:
        out = Path(path) / "page"
        if args.smoke:
            pairs = [
                (plan, old, new)
                for plan, old, new in zip(
                    plans, source["clips"], updated["clips"], strict=True
                )
                if plan.clip_id == args.smoke
            ]
            if not pairs:
                raise ValueError("Smoke clip must belong to the existing page")
            source["clips"] = [old for _, old, _ in pairs]
            updated["clips"] = [new for _, _, new in pairs]
            page["selection"]["included"] = [args.smoke]
        else:
            pairs = list(zip(plans, source["clips"], updated["clips"], strict=True))
        for plan, old_entry, entry in pairs:
            key = plan.clip_id
            for role in ("video", "poster"):
                file = Path(old_entry[role])
                (out / file).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.page / file, out / file)
            if dual_sha256(out / old_entry["video"]) != old_entry["video_sha256"]:
                raise ValueError(f"Stage 1 media hash differs: {key}")
            old = StoreReader(
                args.i2363_run
                / "inputs/dataset"
                / plan.path
                / "annotations/tennis_scene/scene.json"
            )
            candidate = StoreReader(args.candidates / key / "scene.json")
            current = StoreReader(args.scenes / key / "scene.json")
            tracking = StoreReader(args.tracks / key / "scene.json")
            data = load_comparison(old, candidate, plan, pose_record)
            tracks, provenance = load_tracking(tracking, data, track_record)
            pins = current.descriptor(INPUT_NODE)["identity"]["pins"]
            expected = {
                "recipe": RECIPE,
                "source_sha256": provenance["source_sha256"],
                "tracking": provenance["tracking_artifact"],
                "candidates": provenance["candidate_artifact"],
                "alignment": provenance["legacy_camera_alignment"],
                "ball": provenance["legacy_ball"],
                "legacy_scene": provenance["legacy_scene_assembly"],
            }
            if data.provenance != old_entry["provenance"] or any(
                pins[k] != v for k, v in expected.items()
            ):
                raise ValueError(f"Body and comparison inputs differ: {key}")
            bodies = read_bodies(old, root_digest), read_bodies(current, root_digest)
            entry["metrics"] = tracked_clip_metrics(
                old_entry["metrics"],
                tracks,
                track_record["clips"][key]["tracking_and_store_seconds"],
            )
            entry["provenance"] = {
                **provenance,
                "multiview_scene_assembly": current.references["scene_assembly"],
                "placement_frames_by_route": {
                    route: body["player_smpl_valid"].sum(axis=1).tolist()
                    for route, body in zip(("legacy", "multiview"), bodies, strict=True)
                },
            }
            render_clip(
                data,
                plan,
                tracks,
                bodies,
                entry,
                out,
                step,
                crf,
                regressor,
                frame_limit=6 if args.smoke else None,
            )
            for clip in (old_entry, entry):
                check_video(out / clip["video"], clip)
            print(key, entry["video_frames"], entry["video_bytes"], flush=True)
        write_page(out, page)
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        if size > BUNDLE_LIMIT:
            raise ValueError(f"Comparison page exceeds {BUNDLE_LIMIT:,} bytes")
        out.rename(args.output)
    print(f"{args.output}: {len(pairs)} clips, {size:,} bytes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "page",
        "i2363-run",
        "scenes",
        "candidates",
        "pose-record",
        "tracks",
        "track-record",
        "regressor",
        "output",
    ):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument(
        "--smoke", metavar="VIDEO/CLIP", help="Render the first six encoded frames"
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
