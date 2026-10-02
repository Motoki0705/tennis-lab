"""Render saved predictions on exactly the evaluated val JPEGs; CPU only."""

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
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError('Use a new absolute output directory')
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    started = time.monotonic()
    audit = json.loads((BUNDLE / 'collection.json').read_text())
    assert audit['status'] == 'verified'
    plan = json.loads((BUNDLE / 'plan.json').read_text())
    paired = Path(plan['comparison_output'])
    manifest = json.loads((paired / 'manifest.json').read_text())
    assert manifest == json.loads((BUNDLE / 'paired/manifest.json').read_text())
    store = BallFrameStore(Path(manifest['recipe']['store']))
    cache_dir = next(Path(x).parent for x in manifest['input_sha256'] if 'detector-mixed-e9-' in x)
    cache = EvidenceCache(cache_dir, store)
    hashes = dict(manifest['input_sha256'])
    inputs: list[tuple[ClipRecord, ClipEvidence, dict[str, Any]]] = []
    summaries = []
    cameras = ('cam0', 'cam1', 'cam2')
    conditions = ('observed', 'evidence_gap')
    for camera in cameras:
        clip_id = f'meiji/video_000/clip_010/{camera}'
        assert clip_id in manifest['recipe']['partition']['calibration']
        record = store.clip_by_id(clip_id)
        assert record.frame_count == 270 and record.split == 'val'
        evidence = cache.load(clip_id)
        shard = store.directory / 'shards' / shard_name(record.index)
        assert dual_sha256(shard) == hashes[str(shard)]
        values: dict[str, Any] = {}
        for method in ('new_refiner', 'old_refiner'):
            for condition in conditions:
                entry = next(a for a in manifest['artifacts'] if (a['clip_id'], a['method'], a['condition']) == (clip_id, method, condition))
                p = paired / entry['path']
                hashes[str(p)] = entry['sha256']
                assert dual_sha256(p) == entry['sha256']
                with np.load(p, allow_pickle=False) as z:
                    a = {k: z[k] for k in z.files}
                np.testing.assert_equal(a['frame_index'], evidence.frame_index)
                np.testing.assert_equal(a['pts'], evidence.pts)
                values[f'{method}/{condition}'] = a
        if inputs:
            np.testing.assert_equal(evidence.pts, inputs[0][1].pts)
            assert record.time_base == inputs[0][0].time_base and record.fps == inputs[0][0].fps
        reference = values['new_refiner/observed']
        for a in values.values():
            np.testing.assert_equal(a['target_uv'], reference['target_uv'])
            np.testing.assert_equal(a['target_reason'], reference['target_reason'])
        summary: dict[str, Any] = {'clip': clip_id, 'source_size_wh': [record.source_width, record.source_height],
                                   'stored_size_wh': [record.width, record.height], 'media_sha256_inherited': record.media_sha256,
                                   'time_base': record.time_base, 'pts_range': [int(evidence.pts[0]), int(evidence.pts[-1])],
                                   'reasons': {r.name: int((reference['target_reason'] == r).sum()) for r in TargetReason}}
        summary['gap_frames'] = int(values['new_refiner/evidence_gap']['gap_mask'].sum())
        summaries.append(summary)
        inputs.append((record, evidence, values))
    args.output.mkdir(parents=True)
    video = args.output / 'meiji-val-clip010-three-cameras-paired.mp4'
    rate = Fraction(inputs[0][0].fps)
    panel_width, image_height, header = 960, 540, 84
    panel_height = header + image_height
    footer = 126
    width, height = 3 * panel_width, 2 * panel_height + footer
    model_hashes = {
        'new_pilot': audit['training']['best']['checkpoint_sha256'],
        'old_pilot': json.loads((Path(plan['old_run']) / 'best.json').read_text())['checkpoint_sha256'],
        'new_detector': cache.manifest['detector']['sha256'],
        'old_detector': next(v for k, v in hashes.items() if k.endswith('run-i618-convnext-v2-ft-epoch13.ckpt')),
        'new_cache': dual_sha256(cache_dir / 'manifest.json'),
        'old_cache': next(v for k, v in hashes.items() if 'detector-ft-e13-' in k and k.endswith('manifest.json')),
    }
    with av.open(str(video), 'w') as container:
        stream = container.add_stream('libx264', rate=rate)
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        stream.codec_context.thread_count = 2
        stream.options = {'crf': '20', 'preset': 'fast'}
        for frame in range(270):
            canvas: np.ndarray = np.zeros((height, width, 3), np.uint8)
            for column, (record, evidence, values) in enumerate(inputs):
                raw = store.read_bgr(int(store.clip_rows(record)[frame]))
                resized = cv2.resize(raw, (panel_width, image_height), interpolation=cv2.INTER_AREA)
                # Stored labels use source_px * record.scale, not endpoint resizing.
                scale = ((record.source_width - 1) * record.scale * panel_width / record.width,
                         (record.source_height - 1) * record.scale * image_height / record.height)
                for row, condition in enumerate(conditions):
                    x, y = column * panel_width, row * panel_height
                    panel = canvas[y:y + panel_height, x:x + panel_width]
                    image = resized.copy()
                    new, old = [values[f'{method}/{condition}'] for method in ('new_refiner', 'old_refiner')]
                    draw_mixture(image, new['means'][frame], new['scale_tril'][frame], new['mixture_logits'][frame], scale)
                    old_mean, _ = component_geometry(old['means'][frame], old['scale_tril'][frame], old['mixture_logits'][frame], scale)
                    mark(image, old_mean, MAGENTA, cv2.MARKER_TILTED_CROSS)
                    gap = bool(new['gap_mask'][frame])
                    if not gap and bool(evidence.candidates.valid[0, frame, 0]):
                        point = evidence.candidates.coords[0, frame, 0].numpy() * scale
                        mark(image, point, RED, cv2.MARKER_DIAMOND)
                    reason = TargetReason(int(new['target_reason'][frame]))
                    if reason in (TargetReason.OBSERVED, TargetReason.INTERPOLATED, TargetReason.OCCLUSION_ESTIMATED):
                        color = GREEN if reason == TargetReason.OBSERVED else ORANGE
                        mark(image, new['target_uv'][frame] * scale, color, cv2.MARKER_SQUARE)
                    state = reason.name if reason in (TargetReason.OBSERVED, TargetReason.INTERPOLATED, TargetReason.OCCLUSION_ESTIMATED, TargetReason.OUT_OF_FRAME) else f'UNKNOWN:{reason.name}'
                    text_line(panel, f'{record.camera_id}  {condition}  {state}' + ('  ARTIFICIAL GAP' if gap else ''), 12, 22,
                              ORANGE if gap or reason != TargetReason.OBSERVED else GREEN, .58)
                    seconds = float(Fraction(int(evidence.pts[frame])) * Fraction(record.time_base))
                    text_line(panel, f'frame {int(evidence.frame_index[frame]):03d}/269  PTS {int(evidence.pts[frame])} x {record.time_base} = {seconds:.5f}s', 12, 45, size=.52)
                    logits = new['mixture_logits'][frame].astype(np.float64)
                    weights = np.exp(logits - logits.max())
                    weights /= weights.sum()
                    text_line(panel, 'new component 2-sigma ellipse alpha=weight: ' + ', '.join(f'{x:.3f}' for x in weights), 12, 69, size=.49)
                    panel[header:] = image
                    if gap:
                        cv2.rectangle(panel, (1, header), (panel_width - 2, panel_height - 2), ORANGE, 3)
            y = 2 * panel_height
            text_line(canvas, 'Meiji VAL video_000/clip_010 | top: original evidence | bottom: fixed artificial evidence gaps (not RGB occlusion)', 12, y + 23, size=.7)
            text_line(canvas, 'GREEN square: observed label | ORANGE square: estimated label | RED diamond: detector e9 top-1 (decoder order, hidden in gap)', 12, y + 48, size=.65)
            text_line(canvas, 'CYAN +: new mixture mean / weight-alpha component 2-sigma ellipses (NOT 95% HDR) | MAGENTA x: old mixture mean', 12, y + 73, size=.65)
            text_line(canvas, 'SHA256 ' + '  '.join(f'{k}={v[:16]}' for k, v in model_hashes.items()), 12, y + 100, size=.52)
            text_line(canvas, f'Evaluated JPEG input, 59.94006 fps; original frame/PTS above; full hashes and media identity in overlay.json. frame={frame}', 12, y + 120, size=.45)
            encoded = av.VideoFrame.from_ndarray(canvas, format='bgr24')
            encoded.pts, encoded.time_base = frame, 1 / rate
            for packet in stream.encode(encoded):
                container.mux(packet)
            if frame in (0, 40, 80, 120, 160, 200, 240, 269):
                cv2.imwrite(str(args.output / f'frame-{frame:03d}.jpg'), canvas)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(video)) as container:
        decoded = list((f.pts, f.width, f.height) for f in container.decode(video=0))
        assert len(decoded) == 270 and all((w, h) == (width, height) for _, w, h in decoded)
        assert all(a[0] < b[0] for a, b in zip(decoded, decoded[1:], strict=False))
    result = {'status': 'complete', 'video': str(video), 'sha256': dual_sha256(video), 'bytes': video.stat().st_size,
              'frames': 270, 'fps': str(rate), 'size_wh': [width, height], 'seconds': time.monotonic() - started,
              'cpu_threads': {'opencv': 1, 'torch': 1, 'encoder': 2}, 'gpu': False,
              'cameras': summaries, 'model_sha256': model_hashes, 'input_sha256': hashes,
              'point_semantics': 'mixture means in overlay; maximum-weight component means in metric tables',
              'ellipse_semantics': 'each component Mahalanobis radius 2, alpha=weight; not mixture 95% HDR',
              'detector_semantics': 'candidate slot 0 in recorded decoder order, no tie reranking; no point in artificial gaps',
              'timeline': 'all three frame/PTS arrays identical; no resampling; source PTS in caption; native FPS'}
    (args.output / 'overlay.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('video', 'bytes', 'sha256', 'seconds')}))


if __name__ == '__main__':
    main()
