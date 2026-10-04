"""CPU-only play/non-play proposal report. Does not train or run a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from src.tasks.ball_detection.data.play_intervals import (
    PlayIntervalConfig,
    mask_intervals,
)
from src.tasks.ball_detection.data.play_manifest import (
    build_play_manifest,
    clip_evidence,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_detection.review_play_intervals",
    fields=(
        BoundaryPathField("poses", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def render_examples(manifest: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    store = BallFrameStore(Path(manifest["ball_store"]["directory"]))
    chosen: list[dict[str, Any]] = []
    # Include largest non-play proposals and a mostly continuous clip per source.
    # Ranking is selection-only; it never optimizes an evaluation metric.
    for source in ("tracknet", "meiji", "chat_annotation"):
        records = [r for r in manifest["clips"] if r["clip"]["source"] == source and r["clip"]["split"] == "train"]
        ranked = sorted(records, key=lambda r: (
            max((b - a for a, b in r["excluded"]), default=0), r["clip"]["clip_id"],
        ), reverse=True)
        chosen.extend(ranked[:2])
        continuous = sorted(records, key=lambda r: (-r["counts"]["selected_frames"], r["clip"]["clip_id"]))
        if continuous and continuous[0] not in chosen:
            chosen.append(continuous[0])
    fig, axes = plt.subplots(len(chosen), 1, figsize=(14, 2.5 * len(chosen)), squeeze=False)
    examples = []
    for index, record in enumerate(chosen):
        clip = store.clip_by_id(record["clip"]["clip_id"])
        presence, observed, _, times = clip_evidence(store, clip)
        selected = np.zeros(clip.frame_count, bool)
        for a, b in record["play"]:
            selected[a:b] = True
        ax = axes[index, 0]
        dt = float(np.median(np.diff(times)))
        for intervals, y, color in ((record["play"], 0, "#2a9d65"), (record["excluded"], 0, "#d98555"),
                                    (record["training_intervals"], 1, "#9980be"),
                                    (mask_intervals(presence), 2, "#3264b8"), (mask_intervals(observed), 3, "#232323")):
            ax.broken_barh([(float(times[a]), float(times[b - 1] - times[a] + dt)) for a, b in intervals],
                          (y, .7), facecolors=color)
        ax.set(yticks=[.35, 1.35, 2.35, 3.35], yticklabels=["Play / non-play proposal", "Training coverage", "Presence evidence", "Observed GT"],
               xlabel="Seconds (actual PTS)", title=clip.clip_id)
        ax.set_xlim(0, times[-1] + dt)
        # A contact sheet per clip shows selected, bridged and excluded frames.
        groups = (("PLAY", selected), ("GAP IN PLAY", selected & ~presence), ("NON-PLAY?", ~selected))
        sheet = Image.new("RGB", (4 * 320, 3 * 210), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        frame_records = []
        for gi, (name, mask) in enumerate(groups):
            ids = np.flatnonzero(mask)
            if not len(ids):
                draw.text((10, gi * 210 + 12), f"{name}: none", fill="black")
                continue
            samples = ids[np.linspace(0, len(ids) - 1, 4).astype(int)]
            for ci, frame in enumerate(samples):
                row = store.row_of(clip, int(frame))
                rgb = store.read_bgr(row)[..., ::-1].copy()
                im = Image.fromarray(rgb).resize((320, 180))
                painter = ImageDraw.Draw(im)
                inst = store.instances_of(row)
                for xy in inst.xy:
                    if np.isfinite(xy).all():
                        cx, cy = float(xy[0] * 320 / clip.width), float(xy[1] * 180 / clip.height)
                        painter.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline="yellow", width=2)
                sheet.paste(im, (ci * 320, gi * 210))
                label = f"{name} | f{frame} | {times[frame]:.2f}s"
                draw.text((ci * 320 + 4, gi * 210 + 185), label, fill="black")
                frame_records.append(dict(frame=int(frame), seconds=float(times[frame]), proposal=name))
        filename = f"example-{index:02d}.jpg"
        sheet.save(output / filename, quality=88)
        examples.append(dict(clip_id=clip.clip_id, image=filename, frames=frame_records,
                             play=record["play"], excluded=record["excluded"]))
    fig.tight_layout()
    fig.savefig(output / "timelines.png", dpi=110)
    plt.close(fig)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap-seconds", type=float, default=.4)
    parser.add_argument("--min-presence-fraction", type=float, default=.5)
    args = parser.parse_args()
    if not args.poses.is_absolute() or not args.output.is_absolute():
        parser.error("Paths must be absolute")
    roots = RuntimePathRoots(project_root=Path(__file__).resolve().parents[4], data_root=args.poses.parent,
                             checkpoint_root=args.output.parent, artifact_root=args.output.parent,
                             cache_root=args.output.parent, output_root=args.output.parent,
                             external_asset_root=args.output.parent)
    paths = PATH_BOUNDARY.validate({"poses": args.poses, "output": args.output}, resolver=PathResolver(roots))
    args.poses, args.output = paths.declared("poses").path, paths.declared("output").path
    args.output.mkdir(parents=True, exist_ok=False)
    config = PlayIntervalConfig(max_gap_seconds=args.max_gap_seconds,
                                min_presence_fraction=args.min_presence_fraction)
    manifest = build_play_manifest(args.poses, config)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    examples = render_examples(manifest, args.output)
    summary = dict(config=manifest["config"], pose_manifest_sha256=manifest["pose_manifest_sha256"],
                   counts=manifest["counts"], examples=examples,
                   approved_clips=len(manifest["clips"]), excluded_clips=len(manifest["skipped"]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}))


if __name__ == "__main__":
    main()
