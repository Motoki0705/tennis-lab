"""Paired old/new pilot and native detector densities on every validation frame."""

from __future__ import annotations

import gc
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_detection.inference.checkpoint import load_ball_checkpoint
from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tasks.ball_refiner.data.evidence_inference import infer_clip_evidence
from src.tasks.ball_refiner.data.gaps import fixed_gap_mask, validation_partition
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.data.windows import LoadedClip
from src.tasks.ball_refiner.evaluation.detector_selection import validation_clips
from src.tasks.ball_refiner.evaluation.paired_metrics import (
    Array,
    gmm_rows,
    heatmap_rows,
    spatial_reference,
    strata,
    summarize,
)
from src.tasks.ball_refiner.inference import RefinerPair
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tasks.ball_refiner.training.evaluation import predict_clip
from src.tasks.base.training.compilation import compile_modules
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256


@dataclass(frozen=True)
class ComparisonSettings:
    levels: tuple[float, ...]
    samples: int
    seed: int
    chunk_size: int
    uniform_weight: float


def load_pilot(source: Path, device: torch.device) -> tuple[PilotConfig, RefinerPair, dict[str, str]]:
    files = [source / name for name in ('config.yaml', 'data_manifest.json', 'run_state.json', 'best.json')]
    hashes = {str(path): dual_sha256(path) for path in files}
    state, data, best = (json.loads((source / name).read_text()) for name in ('run_state.json', 'data_manifest.json', 'best.json'))
    if state['status'] != 'complete' or best['checkpoint'] != f"epoch-{best['epoch']:03d}.pt":
        raise ValueError('Comparison requires a completed immutable pilot')
    config = PilotConfig.from_config(OmegaConf.load(source / 'config.yaml'))
    checkpoint_path = source / best['checkpoint']
    hashes[str(checkpoint_path)] = dual_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (hashes[str(checkpoint_path)] != best['checkpoint_sha256']
            or checkpoint['data_manifest_sha256'] != hashes[str(source / 'data_manifest.json')]
            or checkpoint['model_config'] != asdict(config.model) or checkpoint['epoch'] != best['epoch']
            or state['best_epoch'] != best['epoch'] or checkpoint['selection_nll_uv'] != best['selection_nll_uv']):
        raise ValueError('Pilot checkpoint/config/selection identity mismatch')
    manifest = config.evidence / 'manifest.json'
    hashes[str(manifest)] = dual_sha256(manifest)
    if hashes[str(manifest)] != data['evidence_manifest_sha256']:
        raise ValueError('Pilot cache changed after training')
    records = validation_clips(BallFrameStore(config.store))
    if validation_partition(tuple(r for r in records if r.source == 'meiji'), config.partition_seed) != data['validation']:
        raise ValueError('Pilot validation partition changed')
    pair = build_ball_refiner_2d(config.model)
    pair.model.load_state_dict(checkpoint['state_dict'], strict=True)
    pair.model.to(device)
    compile_modules({'refiner_2d': pair.model}, config.compilation)
    return config, pair, hashes


def run_comparison(
    old_run: Path, new_run: Path, output: Path, *, settings: ComparisonSettings, device: torch.device,
) -> Path:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json_atomic(output / 'run_state.json', {'status': 'running'})
    rows: dict[str, list[dict[str, Array]]] = defaultdict(list)
    artifacts: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    identities: dict[str, dict[str, str | int]] = {}
    reference: dict[str, Any] | None = None

    def save(method: str, clip: LoadedClip, condition: str, gap: NDArray[np.bool_], values: dict[str, Array], extra: dict[str, Array]) -> None:
        record, target = clip.record, clip.targets
        masks = strata(target, gap, condition)
        groups = [record.source]
        if record.camera_id is not None:
            groups.append(f'{record.source}/{record.camera_id}')
        if record.source == 'meiji':
            groups.append(f"meiji/{'selection' if record.clip_id in partition['selection'] else 'calibration'}")
        for label, selected in masks.items():
            for group in groups:
                rows[f'{method}/{group}/{condition}/{label}'].append({key: value[selected] for key, value in values.items()})
        destination = output / method / f'clip-{record.index:05d}-{condition}.npz'
        destination.parent.mkdir(exist_ok=True)
        with destination.open('xb') as stream:
            np.savez_compressed(stream, **values, **extra, frame_index=target.frame_index, pts=target.pts,
                                target_uv=target.uv, target_reason=target.reason, gap_mask=gap,
                                presence=target.presence, presence_valid=target.presence_valid)
        artifacts.append({'method': method, 'clip_id': record.clip_id, 'source': record.source, 'camera': record.camera_id,
                          'condition': condition, 'frames': record.frame_count, 'path': str(destination.relative_to(output)),
                          'sha256': dual_sha256(destination)})
        write_json_atomic(output / 'progress.json', {'artifacts': artifacts})

    for name, source in (('old', old_run), ('new', new_run)):
        config, pair, hashes = load_pilot(source, device)
        input_hashes.update(hashes)
        store = BallFrameStore(config.store)
        cache = EvidenceCache(config.evidence, store)
        input_hashes.update({str(config.store / filename): digest for filename, digest in cache.manifest['store']['sha256'].items()})
        records = validation_clips(store)
        partition = validation_partition(tuple(r for r in records if r.source == 'meiji'), config.partition_seed)
        recipe = {'model': asdict(config.model), 'training': asdict(config.training), 'seed': config.seed,
                  'window': config.window_length, 'stride': config.stride, 'partition': partition, 'store': str(config.store)}
        if reference is not None and recipe != reference:
            raise ValueError('Comparison pilot recipes or validation frames differ')
        reference = recipe
        for record in records:
            clip = LoadedClip(record, cache.load(record.clip_id), project_store_targets(store, record))
            identity = {'frames': record.frame_count, 'frame_pts_sha256': hashlib.sha256(
                clip.targets.frame_index.tobytes() + clip.targets.pts.tobytes() + clip.targets.reason.tobytes() + clip.targets.uv.tobytes(),
            ).hexdigest()}
            if name == 'new' and identities[record.clip_id] != identity:
                raise ValueError('Paired validation target/frame identity changed')
            identities[record.clip_id] = identity
            gap = fixed_gap_mask(record.frame_count, clip_id=record.clip_id, block_length=config.window_length,
                                 lengths=config.training.gap_lengths, seed=config.partition_seed)
            for condition in ('observed', 'evidence_gap'):
                mask = gap if condition == 'evidence_gap' else np.zeros_like(gap)
                prediction = predict_clip(pair, clip, config, device=device, gap=mask)
                values = gmm_rows(prediction, clip.targets, (record.source_width, record.source_height),
                                  **{key: value for key, value in asdict(settings).items() if key != 'uniform_weight'},
                                  clip_id=record.clip_id, condition=condition, device=device)
                save(f'{name}_refiner', clip, condition, mask, values,
                     {f.name: getattr(prediction, f.name)[0].numpy() for f in fields(prediction)})
            print(json.dumps({'method': f'{name}_refiner', 'clip': record.clip_id}), flush=True)
        del pair
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        detector = cache.manifest['detector']
        checkpoint = Path(detector['checkpoint'])
        if dual_sha256(checkpoint) != detector['sha256']:
            raise ValueError('Detector checkpoint differs from cache')
        input_hashes[str(checkpoint)] = detector['sha256']
        loaded = load_ball_checkpoint(checkpoint, strict=True, weights_only=False)
        predictor = BallDetectionPredictor(loaded.model_io, device, subpixel_refine=detector['subpixel_refine'],
                                           image_normalization=loaded.image_normalization)
        predictor.model.requires_grad_(False)
        for record, cache_record in [(r, next(x for x in cache.manifest['clips'] if x['clip']['clip_id'] == r.clip_id)) for r in records]:
            shard = store.directory / 'shards' / shard_name(record.index)
            if dual_sha256(shard) != cache_record['jpeg_shard_sha256']:
                raise ValueError('Validation JPEG changed after cache generation')
            input_hashes[str(shard)] = cache_record['jpeg_shard_sha256']
            clip = LoadedClip(record, cache.load(record.clip_id), project_store_targets(store, record))
            dense = np.empty((record.frame_count, *clip.evidence.heatmap_size_hw), np.float32)

            def receive(frame: int, heatmap: torch.Tensor, dense: NDArray[np.float32] = dense) -> None:
                dense[frame] = heatmap.numpy()

            evidence = infer_clip_evidence(store, record, predictor, image_size_hw=tuple(detector['image_size_hw']),
                                           stride=detector['stride'], batch_size=detector['batch_size'],
                                           config=cache.candidate_config, heatmap_sink=receive)
            if (not np.array_equal(evidence.window_start, clip.evidence.window_start)
                    or not np.allclose(evidence.argmax_uv, clip.evidence.argmax_uv, atol=1e-5, rtol=1e-5)):
                raise ValueError('Native density replay differs from cached detector/window')
            factor = ((record.width - 1) / record.scale / (record.source_width - 1),
                      (record.height - 1) / record.scale / (record.source_height - 1))
            metrics = heatmap_rows(dense, clip.targets, (record.source_width, record.source_height), factor,
                                   levels=settings.levels, uniform_weight=settings.uniform_weight)
            located = spatial_reference(clip.targets)
            scale = np.array([record.source_width - 1, record.source_height - 1])
            error = np.full(record.frame_count, np.nan)
            error[located] = np.linalg.norm((evidence.argmax_uv[located] - clip.targets.uv[located]) * scale, axis=-1)
            metrics.update(error_px=error, presence_nll=np.full(record.frame_count, np.nan))
            gap = fixed_gap_mask(record.frame_count, clip_id=record.clip_id, block_length=config.window_length,
                                 lengths=config.training.gap_lengths, seed=config.partition_seed)
            save(f'{name}_detector', clip, 'observed', np.zeros_like(gap), metrics,
                 {'argmax_uv': evidence.argmax_uv, 'argmax_score': evidence.argmax_score})
            # Evidence dropout removes the detector's entire spatial observation.
            # Its declared no-evidence density is uniform, with no unique point.
            uniform = {key: value.copy() for key, value in metrics.items()}
            uniform['error_px'][:] = np.nan
            uniform['nll_uv'][located] = 0
            uniform['nll_px'][located] = np.log(scale.prod())
            uniform['coverage'][located] = 1
            uniform['area_px2'][located] = scale.prod()
            save(f'{name}_detector', clip, 'evidence_gap', gap, uniform, {})
            print(json.dumps({'method': f'{name}_detector', 'clip': record.clip_id}), flush=True)
        del predictor, loaded
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if any(dual_sha256(Path(path)) != digest for path, digest in input_hashes.items()):
        raise ValueError('Comparison inputs changed during execution')
    write_json_atomic(output / 'metrics.json', {key: summarize(value, settings.levels) for key, value in sorted(rows.items())})
    write_json_atomic(output / 'manifest.json', {
        'schema': 'ball_refiner_paired_validation.v1', 'settings': asdict(settings), 'input_sha256': input_hashes,
        'artifacts': artifacts, 'paired_frame_identity': identities, 'recipe': reference,
        'scope': 'All three source validation splits; Meiji selection/calibration also separate; no test or fitting',
        'density': 'Native grid Voronoi cells on source UV; fixed uniform component; HDR ties include full tied cells',
        'gmm_hdr': 'R2 GMM HDR with configured fit/area draws; no covariance calibration; maximum-weight mean for point error',
        'gap': 'Fixed candidate-evidence dropout, not RGB occlusion; detector no-evidence prior is uniform, no point',
        'missing': 'Absent: presence NLL only for refiners; unknown has no target; detector scores are not amodal presence',
        'estimated': 'occlusion_estimated/interpolated references reported separately, not independent observed GT',
    })
    write_json_atomic(output / 'run_state.json', {'status': 'complete', 'clips': len(identities), 'files': len(artifacts)})
    return output
