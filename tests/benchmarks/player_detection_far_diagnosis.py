"""#964 four-dev-clip detector diagnosis. GPU inference only via training queue.

No weight, pipeline default, threshold or reserved-test input is modified.
Raw low-score boxes and timings permit CPU-only re-summarization.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.submodules.models import (
    DinoPersonDetector,
    PersonDetectionRequest,
    PersonDetectionResult,
)
from src.submodules.models.dino.architecture import dino_resized_shape
from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tasks.player_detection.evaluation.far_player import far_tiles
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.video import OpenCVVideoFrameReader

DEV_CLIPS = ('video_000/clip_000', 'video_000/clip_007', 'video_001/clip_001', 'video_002/clip_013')
PROBE_SIZES = (1080, 1440, 1800, 2160, 2880, 4320)


def preflight(repo: Path, comparison_path: Path, report: Path) -> dict[str, Any]:
    comparison = json.loads(comparison_path.read_text())
    if comparison['status'] != 'ok' or set(comparison['variants']['player_ft']['clips']) != set(DEV_CLIPS):
        raise ValueError('Require the completed four-dev-clip detector comparison')
    dataset = repo / 'data/tennis_multivew/processed/meiji_3cam/dataset'
    reservation = dataset / 'annotations/player_association/unseen_protocol.json'
    unseen = json.loads(reservation.read_text())
    if set(unseen['clips']).intersection(DEV_CLIPS):
        raise ValueError('Reserved unseen test overlaps requested diagnosis')
    config_path = Path(__file__).resolve().parents[2] / 'src/tennis_scene/configs/pipeline.yaml'
    config = OmegaConf.load(config_path).people_models
    if config.detector != 'dino' or float(config.runtime.dino_detector.confidence) != .3:
        raise ValueError('Expected the unchanged #937 DINO baseline at 0.3')
    weights = repo / 'ckpt' / str(config.dino_checkpoint)
    if dual_sha256(weights) != 'eb8db0ac1b87a0c0a4730534e5dc8107487a53273221ef90231b9351f1d979d4':
        raise ValueError('Expected the immutable #937 exported weights')
    coco = repo / 'ckpt/dino/checkpoint0029_4scale_swin.pth'
    records = []
    for clip in DEV_CLIPS:
        label_file = label_path(dataset, clip)
        labels = ClipLabels.load(label_file)
        label_hash = dual_sha256(label_file)
        prior_hash = comparison['labels'][clip]['sha256']
        if label_hash != prior_hash and labels.provenance.get('migration', {}).get('source_label_sha256') != prior_hash:
            raise ValueError(f'Dev labels changed since baseline: {clip}')
        root = Path(comparison['variants']['player_ft']['clips'][clip]['store'])
        source = json.loads((root / 'scene.json').read_text())['source']
        store = ClipStore(root, source)
        reference = store.active('court_calibration')
        if reference is None:
            raise ValueError(f'Missing baseline calibration: {clip}')
        calibration = store.load(reference, ArtifactCodec(CourtCalibrationOutput))
        for video in source['videos']:
            camera = video['camera_id']
            if video['num_frames'] != labels.num_frames or (video['width'], video['height']) != (1920, 1080):
                raise ValueError('Expected complete 1920x1080 Meiji timeline')
            if dual_sha256(Path(video['path'])) != video['sha256']:
                raise ValueError('Dev video content changed')
            detection_ref = store.active(f'person_detection/{camera}')
            if detection_ref is None or calibration.footpoint_polygons[camera] is None:
                raise ValueError(f'Missing baseline detector/ROI: {clip}/{camera}')
            detection = store.load(detection_ref, ArtifactCodec(PersonDetectionOutput))
            if len(detection.frame_offsets) != labels.num_frames + 1:
                raise ValueError('Baseline detector timeline differs')
            records.append({'clip': clip, 'camera': camera, 'video': video,
                'label_path': str(label_file), 'label_sha256': label_hash,
                'baseline_store': str(root), 'baseline_reference': json_value(detection_ref),
                'roi': json_value(calibration.footpoint_polygons[camera]), 'calibration_reference': json_value(reference)})
    return {'schema': 'far_player_diagnosis_v1', 'status': 'planned',
        'comparison': str(comparison_path), 'comparison_sha256': dual_sha256(comparison_path),
        'reservation': str(reservation), 'reservation_sha256': dual_sha256(reservation),
        'pipeline_config_sha256': dual_sha256(config_path),
        'weights': {'ft': {'path': str(weights), 'sha256': dual_sha256(weights)},
                    'coco': {'path': str(coco), 'sha256': dual_sha256(coco)}},
        'repository': str(repo / 'third_party' / str(config.dino_repository)),
        'baseline_size': [int(config.runtime.dino_detector.short_side), int(config.runtime.dino_detector.max_long_side)],
        'floor': .01, 'thresholds': [.01, .05, .1, .2, .3, .5], 'agreement_iou': .5, 'score_overlap_iou': .3,
        'capacity_probe_short_sides': list(PROBE_SIZES), 'tile_size': [1080, 1920],
        'tiles': json_value(far_tiles(1920, 1080)), 'union_max_height_at_1080p': 64, 'dedup_iou': .5,
        'scope': 'same_saved_court_roi', 'interpretation': 'old_COCO_box_agreement_not_detection_recall',
        'near_far': 'exactly_two_player_box_bottom_rank; nonplayers use midpoint of those bottoms; otherwise unknown',
        'runtime_scope': 'synchronized predictor (CPU resize + GPU + decode) and CPU ROI/union; '
                         'excludes video decode, model load, warmup, archive IO; same camera ms/frame repeated by near/far',
        'report': str(report), 'inputs': records, 'archives': {}}


def predict_timed(detector: DinoPersonDetector, image: NDArray[np.uint8]) -> tuple[PersonDetectionResult, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = detector.predict(PersonDetectionRequest(image))
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def probe(detector: DinoPersonDetector, images: list[NDArray[np.uint8]], report: Path) -> dict[str, Any]:
    trials = []
    largest: int | None = None
    for short in PROBE_SIZES:
        detector.short_side, detector.max_long_side = short, math.ceil(short * 1920 / 1080)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trial: dict[str, Any] = {'short_side': short, 'max_long_side': detector.max_long_side,
            'input_hw': list(dino_resized_shape(1080, 1920, short_side=short, max_long_side=detector.max_long_side))}
        try:
            for image in images:
                predict_timed(detector, image)
            trial['status'] = 'ok'
            largest = short
        except torch.cuda.OutOfMemoryError as error:
            trial.update(status='oom', error=str(error))
        trial['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        trials.append(trial)
        write_json_atomic(report / 'capacity_probe.json', {'trials': trials, 'largest_successful_short_side': largest})
        gc.collect()
        torch.cuda.empty_cache()
        if trial['status'] != 'ok':
            break
    if largest is None:
        raise RuntimeError('Native 1080 inference did not fit; diagnosis cannot substitute another size')
    result = {'trials': trials, 'largest_successful_short_side': largest,
        'definition': 'largest successful tested size on all three dev-camera first frames; hardware/precision specific, '
                      'not an architectural maximum or a deployment choice',
        'gpu': torch.cuda.get_device_name(), 'precision': 'float32', 'batch_size': 1}
    write_json_atomic(report / 'capacity_probe.json', result)
    return result


def infer_camera(detector: DinoPersonDetector, record: dict[str, Any], *, tiled: bool) -> DetectionArchive:
    offsets, boxes, scores, times = [0], [], [], []
    video = record['video']
    if dual_sha256(Path(video['path'])) != video['sha256']:
        raise ValueError('Video changed after preflight')
    for packet in OpenCVVideoFrameReader(Path(video['path']), max_frames=video['num_frames']):
        if tiled:
            parts, elapsed = [], 0.
            for x1, y1, x2, y2 in far_tiles(video['width'], video['height']):
                result, milliseconds = predict_timed(detector, packet.frame[y1:y2, x1:x2])
                mapped = result.boxes_xyxy + np.asarray([x1, y1, x1, y1], np.float32)
                parts.append(PersonDetectionResult(mapped, result.scores))
                elapsed += milliseconds
            prediction = PersonDetectionResult(np.concatenate([p.boxes_xyxy for p in parts]),
                                               np.concatenate([p.scores for p in parts]))
        else:
            prediction, elapsed = predict_timed(detector, packet.frame)
        offsets.append(offsets[-1] + len(prediction.scores))
        boxes.append(prediction.boxes_xyxy)
        scores.append(prediction.scores)
        times.append(elapsed)
    if len(times) != video['num_frames']:
        raise ValueError('Incomplete raw video decoding')
    return DetectionArchive(np.asarray(offsets, np.int64), np.concatenate(boxes), np.concatenate(scores), np.asarray(times, np.float64))


def infer(plan: dict[str, Any], report: Path) -> None:
    if (report / 'progress.json').exists() or (report / 'inference.json').exists():
        raise FileExistsError('Do not overwrite any previous inference, including a failed run')
    plan['status'] = 'running'
    write_json_atomic(report / 'progress.json', plan)
    first_clip = [r for r in plan['inputs'] if r['clip'] == DEV_CLIPS[0]]
    images = [next(iter(OpenCVVideoFrameReader(Path(r['video']['path']), max_frames=1))).frame for r in first_clip]
    try:
        for weight_name in ('ft', 'coco'):
            weight = plan['weights'][weight_name]
            if dual_sha256(Path(weight['path'])) != weight['sha256']:
                raise ValueError('Weights changed after preflight')
            detector = DinoPersonDetector(Path(weight['path']), Path(plan['repository']), device='cuda',
                confidence=.01 if weight_name == 'ft' else .3,
                short_side=plan['baseline_size'][0], max_long_side=plan['baseline_size'][1])
            try:
                detector.load()
                if weight_name == 'ft':
                    plan['capacity'] = probe(detector, images, report)
                    largest = plan['capacity']['largest_successful_short_side']
                    specs = [('ft_base', *plan['baseline_size'], False), ('ft_1080', 1080, 1920, False)]
                    if largest > 1080:
                        specs.append(('ft_largest', largest, math.ceil(largest * 1920 / 1080), False))
                    specs.append(('ft_tiles', *plan['tile_size'], True))
                else:
                    specs = [('coco_base', *plan['baseline_size'], False)]
                for name, short, long, tiled in specs:
                    detector.short_side, detector.max_long_side = short, long
                    # Warmup is separate from every reported per-frame sample.
                    predict_timed(detector, images[0][:648, :1056] if tiled else images[0])
                    records = plan['archives'][name] = {}
                    for record in plan['inputs']:
                        key = f"{record['clip']}/{record['camera']}"
                        plan['current'] = {'variant': name, 'key': key}
                        write_json_atomic(report / 'progress.json', plan)
                        archive = infer_camera(detector, record, tiled=tiled)
                        records[key] = archive.save(report / 'raw' / name / f'{key}.npz')
                        write_json_atomic(report / 'progress.json', plan)
                        print(json.dumps({'variant': name, 'key': key, **records[key]}), flush=True)
            finally:
                detector.unload()
                del detector
                gc.collect()
                torch.cuda.empty_cache()
    except Exception as error:
        plan.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json_atomic(report / 'progress.json', plan)
        raise
    plan['status'] = 'ok'
    write_json_atomic(report / 'inference.json', plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'infer', 'summarize'), required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    if args.phase == 'preflight':
        report.mkdir(parents=True, exist_ok=True)
        if (report / 'plan.json').exists():
            raise FileExistsError(report / 'plan.json')
        plan = preflight(args.repo.resolve(), args.comparison.resolve(), report)
        write_json_atomic(report / 'plan.json', plan)
        print(json.dumps({'inputs': len(plan['inputs']), 'frames': sum(r['video']['num_frames'] for r in plan['inputs'])}))
    elif args.phase == 'infer':
        planned = json.loads((report / 'plan.json').read_text())
        if preflight(args.repo.resolve(), args.comparison.resolve(), report) != planned:
            raise ValueError('Preflight inputs/settings changed; use a new run')
        infer(planned, report)
        from src.tasks.player_detection.evaluation.far_report import summarize_report
        summarize_report(report)
    else:
        from src.tasks.player_detection.evaluation.far_report import summarize_report
        summarize_report(report)


if __name__ == '__main__':
    main()
