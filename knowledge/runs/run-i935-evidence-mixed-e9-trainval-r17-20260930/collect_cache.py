"""CPU collection of the fixed epoch-9 cache and paired validation recall."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.evaluation.candidate_recall import candidate_recall_counts
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.evaluation.detector_selection import summarize_clips
from src.utils.checksum import dual_sha256

ROOT = Path('/home/kamimura/projects/tennis-lab')
BUNDLE = Path(__file__).resolve().parent
PREVIOUS = BUNDLE.parent / 'run-i935-mixed-ft-val-recall-s42-r16-20260929'
JOB = '1790725512433633211_3869834_i935-evidence-mixed-e9-trainval-r17-20260930'


def main() -> None:
    torch.set_num_threads(4)
    plan_path = PREVIOUS / 'cache_plan.json'
    plan = json.loads(plan_path.read_text())
    report = Path(plan['report'])
    resource = json.loads((report / 'resource_usage.json').read_text())
    hashes = json.loads((report / 'artifact_hashes.json').read_text())
    queue = ROOT / '.training_queue'
    state = (queue / 'state' / f'{JOB}.state').read_text()
    if not (queue / 'done' / f'{JOB}.job').is_file() or 'state=done\n' not in state:
        raise ValueError('Queue did not publish successful completion')
    if resource['status'] != 'complete' or resource['queue_job'] != JOB:
        raise ValueError('Resource report does not confirm this job completed')
    for path, expected in plan['input_sha256'].items():
        if dual_sha256(Path(path)) != expected:
            raise ValueError(f'Fixed input changed: {path}')
    store = BallFrameStore(Path(plan['store']))
    new = EvidenceCache(Path(plan['output']), store)
    old = EvidenceCache(Path(plan['reference_manifest']).parent, store)
    manifest = new.directory / 'manifest.json'
    if dual_sha256(manifest) != hashes['manifest_sha256'] or manifest.read_bytes() != (report / 'manifest.json').read_bytes():
        raise ValueError('Live/snapshot manifest or checksum disagrees')
    if hashes['plan_sha256'] != dual_sha256(plan_path) or hashes['checkpoint_sha256'] != new.manifest['detector']['sha256']:
        raise ValueError('Plan or checkpoint identity disagrees')
    if new.manifest['selection'] != old.manifest['selection'] or new.manifest['context'] != {'pose': 'not_generated', 'court': 'not_generated'}:
        raise ValueError('Unexpected cache coverage or context')
    for key, expected in plan['settings'].items():
        if new.manifest['detector'][key] != expected:
            raise ValueError(f'Inference recipe changed: {key}')
    indexed = {item['path']: item for item in hashes['files']}
    paths = {str(p) for p in new.directory.rglob('*.npz')}
    if len(indexed) != len(hashes['files']) or set(indexed) != paths:
        raise ValueError('artifact_hashes does not cover every NPZ exactly once')
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {'clips': 0, 'frames': 0})
    evaluated: dict[str, list[dict[str, Any]]] = {'epoch9_cache': [], 'ft_e13_cache': []}
    for record, old_record in zip(new.manifest['clips'], old.manifest['clips'], strict=True):
        if record['clip'] != old_record['clip'] or record['jpeg_shard_sha256'] != old_record['jpeg_shard_sha256']:
            raise ValueError('New and old clip identities disagree')
        path = new.directory / record['file']
        saved = indexed[str(path)]
        if saved['sha256'] != record['sha256'] or dual_sha256(path) != saved['sha256'] or path.stat().st_size != saved['bytes']:
            raise ValueError(f'Artifact checksum/size mismatch: {path}')
        clip = store.clip_by_id(record['clip']['clip_id'])
        evidence = new.load(clip.clip_id)
        if len(evidence.frame_index) != clip.frame_count:
            raise ValueError('Reader frame count mismatch')
        group = groups[f'{clip.source}/{clip.split}']
        group['clips'] += 1
        group['frames'] += clip.frame_count
        if clip.split != 'val':
            continue
        targets = project_store_targets(store, clip)
        scale = np.asarray([clip.source_width - 1, clip.source_height - 1], np.float32)
        for name, item in [('epoch9_cache', evidence), ('ft_e13_cache', old.load(clip.clip_id))]:
            counts = candidate_recall_counts(
                item.candidates.coords[0].numpy() * scale, item.candidates.scores[0].numpy(),
                item.candidates.valid[0].numpy(), targets.uv * scale, targets.position_valid, radius_px=20,
            )
            evaluated[name].append({'clip': record['clip'], 'counts': asdict(counts)})
    clips, frames = sum(x['clips'] for x in groups.values()), sum(x['frames'] for x in groups.values())
    if (clips, frames) != (329, 145767) or (clips, frames) != (hashes['validated_clips'], hashes['validated_frames']):
        raise ValueError('Plan, generator audit and fresh reader counts disagree')
    if sum(x['bytes'] for x in hashes['files']) != hashes['files_bytes']:
        raise ValueError('Artifact byte totals disagree')
    summaries = {name: summarize_clips(rows) for name, rows in evaluated.items()}
    bf16 = {row['group']: row for row in json.loads((PREVIOUS / 'per_epoch.json').read_text()) if row['epoch'] == 9}
    comparison = []
    for group in summaries['epoch9_cache']:
        current, reference = summaries['epoch9_cache'][group], bf16[group]
        if current['observed'] != reference['observed']:
            raise ValueError(f'Validation denominators differ: {group}')
        comparison.append({'group': group, 'epoch9_cache': current, 'ft_e13_cache': summaries['ft_e13_cache'][group],
                           'epoch9_bf16_validation': reference,
                           'delta_recall8_pp': 100 * (current['recall_at_k'] - reference['recall_at_8_20px']),
                           'delta_recall1_pp': 100 * (current['recall_at_1'] - reference['recall_at_1_20px'])})
    audit = {'status': 'passed', 'queue_job': JOB, 'queue_state': state, 'queue_exit_code': 0,
             'exit_evidence': 'done job + state=done; queue wrapper only publishes done when wait returns exit code 0',
             'clips': clips, 'frames': frames, 'groups': dict(groups), 'all_clip_reader': 'passed',
             'all_npz_sha256_and_size': 'passed', 'npz_bytes': hashes['files_bytes'],
             'manifest_sha256': hashes['manifest_sha256'], 'checkpoint_sha256': hashes['checkpoint_sha256'],
             'resource_usage': resource, 'comparisons': comparison,
             'limitations': ['No test inference or scoring', 'No original media/JPEG rehash in this collection',
                             'PyTorch allocated/reserved statistics are not whole-device VRAM']}
    for name in ['manifest.json', 'artifact_hashes.json', 'resource_usage.json', 'preflight.json']:
        if (report / name).is_file():
            shutil.copyfile(report / name, BUNDLE / name)
    shutil.copyfile(queue / 'done' / f'{JOB}.job', BUNDLE / 'done.job')
    shutil.copyfile(queue / 'state' / f'{JOB}.state', BUNDLE / 'queue.state')
    shutil.copyfile(queue / 'logs' / f'{JOB}.log', BUNDLE / 'queue.log')
    for name, value in [('collection.json', audit), ('recall_clips.json', evaluated)]:
        (BUNDLE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    lines = ['| source/camera | observed | new @8 / @1 | old @8 / @1 | bf16 @8 / @1 | new−bf16 pp @8 / @1 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in comparison:
        new_row, old_row, bf = row['epoch9_cache'], row['ft_e13_cache'], row['epoch9_bf16_validation']
        lines.append(f"| {row['group']} | {new_row['observed']} | {new_row['recall_at_k']:.6f} / {new_row['recall_at_1']:.6f} | "
                     f"{old_row['recall_at_k']:.6f} / {old_row['recall_at_1']:.6f} | {bf['recall_at_8_20px']:.6f} / {bf['recall_at_1_20px']:.6f} | "
                     f"{row['delta_recall8_pp']:+.4f} / {row['delta_recall1_pp']:+.4f} |")
    (BUNDLE / 'recall.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'clips': clips, 'frames': frames, 'groups': dict(groups), 'status': 'passed'}))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
