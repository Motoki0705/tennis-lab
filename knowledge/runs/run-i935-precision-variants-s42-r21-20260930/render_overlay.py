"""Run-19 JPEG/PTS renderer adapted to the anchored variant; CPU only."""

from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import torch

from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord, shard_name
from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tasks.ball_refiner.data.targets import TargetReason
from src.tasks.ball_refiner.visualization.overlay import (
    GREEN,
    MAGENTA,
    ORANGE,
    RED,
    component_geometry,
    draw_mixture,
    mark,
    text_line,
)
from src.utils.checksum import dual_sha256

BUNDLE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip", choices=("clip010", "hard"), required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError("Use a new absolute output directory")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    started = time.monotonic()
    audit = json.loads((BUNDLE / "collection.json").read_text())
    assert audit["status"] == "verified"
    plan = json.loads((BUNDLE / "plan.json").read_text())
    paired = Path(plan["reference"])
    manifest = json.loads((paired / "manifest.json").read_text())
    variant = next(v for v in plan["variants"] if v["name"] == "anchored_12k")
    variant_dir = Path(variant["evaluation_output"])
    vm = json.loads((variant_dir / "manifest.json").read_text())
    assert vm == json.loads((BUNDLE / "evaluation-anchored_12k/manifest.json").read_text())
    selected = json.loads((BUNDLE / "clip_selection.json").read_text())["selected"]
    clip, start, length = ("meiji/video_000/clip_010", 0, 270) if args.clip == "clip010" else (
        selected["clip"], selected["start"], selected["length"])
    assert clip.startswith("meiji/video_000/")
    store = BallFrameStore(Path(manifest["recipe"]["store"]))
    cache_dir = next(Path(x).parent for x in manifest["input_sha256"] if "detector-mixed-e9-" in x)
    cache = EvidenceCache(cache_dir, store)
    hashes = {str(cache_dir / "manifest.json"): dual_sha256(cache_dir / "manifest.json")}
    inputs: list[tuple[ClipRecord, ClipEvidence, dict[str, Any]]] = []
    summaries = []
    conditions = ("observed", "evidence_gap")
    for camera in ("cam0", "cam1", "cam2"):
        clip_id = f"{clip}/{camera}"
        record = store.clip_by_id(clip_id)
        assert record.split == "val" and start + length <= record.frame_count
        if args.clip == "clip010":
            assert record.frame_count == 270
        evidence = cache.load(clip_id)
        shard = store.directory / "shards" / shard_name(record.index)
        assert dual_sha256(shard) == manifest["input_sha256"][str(shard)]
        hashes[str(shard)] = dual_sha256(shard)
        values: dict[str, Any] = {}
        for method in ("anchored_12k", "r18", "detector"):
            for condition in conditions:
                if method == "anchored_12k":
                    entry = next(a for a in vm["artifacts"] if (a["clip_id"], a["condition"]) == (clip_id, condition))
                    path = variant_dir / entry["path"]
                else:
                    ref_method = "new_refiner" if method == "r18" else "new_detector"
                    entry = next(a for a in manifest["artifacts"] if (a["clip_id"], a["method"], a["condition"]) == (clip_id, ref_method, condition))
                    path = paired / entry["path"]
                hashes[str(path)] = entry["sha256"]
                assert dual_sha256(path) == entry["sha256"]
                with np.load(path, allow_pickle=False) as z:
                    arrays = dict(z)
                np.testing.assert_equal(arrays["frame_index"], evidence.frame_index)
                np.testing.assert_equal(arrays["pts"], evidence.pts)
                values[f"{method}/{condition}"] = arrays
        if inputs:
            np.testing.assert_equal(evidence.pts, inputs[0][1].pts)
            assert record.time_base == inputs[0][0].time_base and record.fps == inputs[0][0].fps
        reference = values["anchored_12k/observed"]
        for arrays in values.values():
            for key in ("target_uv", "target_reason", "presence", "presence_valid"):
                np.testing.assert_equal(arrays[key], reference[key])
        for condition in conditions:
            for method in ("r18", "detector"):
                np.testing.assert_equal(values[f"{method}/{condition}"]["gap_mask"], values[f"anchored_12k/{condition}"]["gap_mask"])
        # Verify the drawn detector slot0 agrees with the error used to choose the clip.
        known = reference["target_reason"] == TargetReason.OBSERVED
        scale_source = np.array([record.source_width - 1, record.source_height - 1])
        det_error = np.linalg.norm((evidence.candidates.coords[0, :, 0].numpy() - reference["target_uv"]) * scale_source, axis=-1)
        valid = known & evidence.candidates.valid[0, :, 0].numpy()
        np.testing.assert_allclose(det_error[valid], values["detector/observed"]["error_px"][valid], rtol=1e-5, atol=1e-4)
        observed = known.copy()
        observed[:start] = False
        observed[start + length:] = False
        wrong = observed & (values["detector/observed"]["error_px"] > 20)
        fixed = wrong & (reference["error_px"] <= 20)
        failed = wrong & (reference["error_px"] > 20)
        worsened = observed & (values["detector/observed"]["error_px"] <= 20) & (reference["error_px"] > 20)
        summary: dict[str, Any] = {
            "clip": clip_id, "source_size_wh": [record.source_width, record.source_height],
            "stored_size_wh": [record.width, record.height], "media_sha256_inherited": record.media_sha256,
            "time_base": record.time_base, "pts_range": [int(evidence.pts[start]), int(evidence.pts[start + length - 1])],
            "reasons": {r.name: int((reference["target_reason"][start:start + length] == r).sum()) for r in TargetReason},
            "gap_frames": int(values["anchored_12k/evidence_gap"]["gap_mask"][start:start + length].sum()),
            "wrong_detector_frames": np.flatnonzero(wrong).tolist(),
            "corrected_frames": np.flatnonzero(fixed).tolist(), "remaining_failure_frames": np.flatnonzero(failed).tolist(),
            "introduced_error_frames": np.flatnonzero(worsened).tolist(),
        }
        summaries.append(summary)
        inputs.append((record, evidence, values))
    args.output.mkdir(parents=True)
    video = args.output / f"{clip.split('/')[-1]}-frames-{start:04d}-{start + length - 1:04d}-anchored-vs-r18.mp4"
    rate = Fraction(inputs[0][0].fps)
    panel_width, image_height, header = 960, 540, 108
    panel_height, footer = header + image_height, 148
    width, height = 3 * panel_width, 2 * panel_height + footer
    model_hashes = {
        "anchored_12k": audit["variants"]["anchored_12k"]["best"]["checkpoint_sha256"],
        "r18": json.loads((Path(plan["baseline"]) / "best.json").read_text())["checkpoint_sha256"],
        "detector_e9": cache.manifest["detector"]["sha256"],
        "cache": dual_sha256(cache_dir / "manifest.json"),
    }
    snapshots = {start, start + length - 1, start + length // 2}
    for summary in summaries:
        for key in ("corrected_frames", "remaining_failure_frames", "introduced_error_frames"):
            if summary[key]:
                snapshots.add(summary[key][0])
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.codec_context.thread_count = 2
        stream.options = {"crf": "20", "preset": "fast"}
        for offset, frame in enumerate(range(start, start + length)):
            canvas: np.ndarray = np.zeros((height, width, 3), np.uint8)
            for column, (record, evidence, values) in enumerate(inputs):
                raw = store.read_bgr(int(store.clip_rows(record)[frame]))
                resized = cv2.resize(raw, (panel_width, image_height), interpolation=cv2.INTER_AREA)
                scale = ((record.source_width - 1) * record.scale * panel_width / record.width,
                         (record.source_height - 1) * record.scale * image_height / record.height)
                for row, condition in enumerate(conditions):
                    x, y = column * panel_width, row * panel_height
                    panel = canvas[y:y + panel_height, x:x + panel_width]
                    image = resized.copy()
                    new, old = [values[f"{method}/{condition}"] for method in ("anchored_12k", "r18")]
                    draw_mixture(image, new["means"][frame], new["scale_tril"][frame], new["mixture_logits"][frame],
                                 scale, point_summary="top_component")
                    old_mean, _ = component_geometry(old["means"][frame], old["scale_tril"][frame], old["mixture_logits"][frame], scale)
                    mark(image, old_mean, MAGENTA, cv2.MARKER_TILTED_CROSS)
                    gap = bool(new["gap_mask"][frame])
                    if not gap and bool(evidence.candidates.valid[0, frame, 0]):
                        mark(image, evidence.candidates.coords[0, frame, 0].numpy() * scale, RED, cv2.MARKER_DIAMOND)
                    reason = TargetReason(int(new["target_reason"][frame]))
                    if reason in (TargetReason.OBSERVED, TargetReason.INTERPOLATED, TargetReason.OCCLUSION_ESTIMATED):
                        color = GREEN if reason == TargetReason.OBSERVED else ORANGE
                        mark(image, new["target_uv"][frame] * scale, color, cv2.MARKER_SQUARE)
                    state = reason.name if reason in (TargetReason.OBSERVED, TargetReason.INTERPOLATED, TargetReason.OCCLUSION_ESTIMATED, TargetReason.OUT_OF_FRAME) else f"UNKNOWN:{reason.name}"
                    text_line(panel, f"{record.camera_id} {condition} {state}" + (" ARTIFICIAL GAP" if gap else ""), 12, 22,
                              ORANGE if gap or reason != TargetReason.OBSERVED else GREEN, .58)
                    seconds = float(Fraction(int(evidence.pts[frame])) * Fraction(record.time_base))
                    text_line(panel, f"frame {int(evidence.frame_index[frame]):04d}/{record.frame_count - 1}  PTS {int(evidence.pts[frame])} x {record.time_base} = {seconds:.5f}s", 12, 45, size=.52)
                    logits = new["mixture_logits"][frame].astype(np.float64)
                    weights = np.exp(logits - logits.max())
                    weights /= weights.sum()
                    text_line(panel, "anchored component 2-sigma alpha=weight: " + ", ".join(f"{p:.3f}" for p in weights), 12, 69, size=.49)
                    if reason == TargetReason.OBSERVED:
                        detector_text = "hidden" if gap else f"{values['detector/observed']['error_px'][frame]:.1f}px"
                        text_line(panel, f"Observed error: detector {detector_text}; anchored top component {new['error_px'][frame]:.1f}px", 12, 92, size=.51)
                    else:
                        text_line(panel, "Estimated/unknown label: not an observed accuracy measurement", 12, 92, ORANGE, .51)
                    panel[header:] = image
                    if gap:
                        cv2.rectangle(panel, (1, header), (panel_width - 2, panel_height - 2), ORANGE, 3)
            y = 2 * panel_height
            text_line(canvas, f"{clip} VAL | top: original evidence | bottom: fixed artificial evidence gaps (not RGB occlusion)", 12, y + 23, size=.7)
            text_line(canvas, "GREEN square: observed label | ORANGE square: estimated label | RED diamond: e9 top-1 (decoder order, hidden in gaps)", 12, y + 48, size=.65)
            text_line(canvas, "CYAN +: anchored_12k TOP COMPONENT mean; ellipses: component 2-sigma, alpha=weight (NOT mixture HDR95) | MAGENTA x: r18 MIXTURE mean", 12, y + 73, size=.60)
            text_line(canvas, "SHA256 " + "  ".join(f"{key}={value[:16]}" for key, value in model_hashes.items()), 12, y + 100, size=.55)
            text_line(canvas, f"Evaluated JPEG / native FPS {float(rate):.5f} / original frame and PTS above / full hashes in overlay.json / frame={frame}", 12, y + 125, size=.50)
            encoded = av.VideoFrame.from_ndarray(canvas, format="bgr24")
            encoded.pts, encoded.time_base = offset, 1 / rate
            for packet in stream.encode(encoded):
                container.mux(packet)
            if frame in snapshots:
                cv2.imwrite(str(args.output / f"frame-{frame:04d}.jpg"), canvas)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(video)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        decoded = []
        for offset, frame_image in enumerate(container.decode(video=0)):
            decoded.append((frame_image.pts, frame_image.width, frame_image.height))
            if offset == length // 2:
                cv2.imwrite(str(args.output / "decoded-middle.jpg"), frame_image.to_ndarray(format="bgr24"))
        assert len(decoded) == length and all((w, h) == (width, height) for _, w, h in decoded)
        assert all(a[0] < b[0] for a, b in zip(decoded, decoded[1:], strict=False))
    r19_script = BUNDLE.parent / "run-i935-detector-only-mixed-e9-s42-r18-20260930/render_overlay.py"
    result = {"status": "complete", "video": str(video), "sha256": dual_sha256(video), "bytes": video.stat().st_size,
              "clip": clip, "start": start, "frames": length, "fps": str(rate), "size_wh": [width, height],
              "seconds": time.monotonic() - started, "cpu_threads": {"opencv": 1, "torch": 1, "encoder": 2}, "gpu": False,
              "cameras": summaries, "model_sha256": model_hashes, "input_sha256": hashes,
              "renderer_sha256": dual_sha256(Path(__file__)), "r19_renderer_sha256": dual_sha256(r19_script),
              "point_semantics": "r18 mixture mean; anchored maximum-weight component mean; tables use maximum-weight component for both",
              "ellipse_semantics": "component Mahalanobis radius 2; alpha=weight; not mixture 95% HDR",
              "timeline": "all 3 camera frame/PTS identical; native FPS; no resampling; source PTS in caption",
              "selection": selected if args.clip == "hard" else "directive clip010, all 270 frames"}
    (args.output / "overlay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("video", "sha256", "bytes", "seconds")}, indent=2))


if __name__ == "__main__":
    main()

