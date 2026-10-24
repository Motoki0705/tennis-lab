"""Validate measured pilot receipts and summarize without filling missing values."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from pilot import CONDITIONS
from session_guard import write


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def measured_row(entry: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = dict(condition=entry['condition'], status=entry['status'],
                               optimizer_updates=0 if result and not result['records'] else None,
                               first_step_seconds=None, step_mean_seconds=None, warm_step_mean_seconds=None,
                               warm_step_median_seconds=None, data_wait_mean_seconds=None,
                               peak_allocated_bytes=None, peak_reserved_bytes=None,
                               evaluation_seconds=None, local_serialization_seconds=None,
                               local_hash_seconds=None, checkpoint_publication_mean_seconds=None,
                               condition_wall_seconds=entry.get('seconds'))
    if result is None:
        row['reason'] = entry.get('reason', 'No completed condition receipt')
        return row
    records = result['records']
    if len(records) > 10 or [r['update'] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError('Pilot updates exceed their bound or are not contiguous')
    if entry.get('seconds', 0) > 1200:
        raise ValueError('Condition exceeded the authorized 20-minute bound')
    if result['status'] != entry['status']:
        raise ValueError('Condition receipt and supervisor status differ')
    if entry['status'] == 'completed':
        if len(records) != 10 or not result['completed_marker'] or result['latest']['update'] != 10:
            raise ValueError('A complete pilot must have ten updates and a final checkpoint')
        checkpoints = [p for p in result['publications'] if p['stage'] == 'checkpoint']
        if [p['update'] for p in checkpoints] != [5, 10] or any(p['status'] != 'verified' for p in result['publications']):
            raise ValueError('Completed condition lacks verified checkpoint publications')
    if result['architecture'] != entry['condition'].split('-T')[0] or result['gpu']['name'] != 'NVIDIA L4':
        raise ValueError('Declared architecture or observed GPU differs from the condition')
    model, runtime = result['model'], result['runtime']
    if (model['frames'] != int(entry['condition'].split('-T')[1]) or model['input_channels'] != 5
            or model['decoder_channels'] != 128 or model['frame_chunk_size'] != 4
            or not model['activation_checkpointing'] or runtime['precision'] != 'bf16'
            or runtime['compile_mode'] != 'off' or runtime['jpeg_decoder'] != 'nvjpeg'):
        raise ValueError('Measured model or runtime differs from the fixed pilot')
    if any(r['input_frames'] != 128 or r['microbatches'] != 128 // model['frames'] for r in records):
        raise ValueError('Input-frame or accumulation budget differs between conditions')
    if records:
        row.update(optimizer_updates=len(records), first_step_seconds=records[0]['seconds'],
                   step_mean_seconds=mean([r['seconds'] for r in records]),
                   warm_step_mean_seconds=mean([r['seconds'] for r in records[1:]]),
                   warm_step_median_seconds=statistics.median([r['seconds'] for r in records[1:]]) if len(records) > 1 else None,
                   data_wait_mean_seconds=mean([r['data_wait_seconds'] for r in records]),
                   peak_allocated_bytes=max(r['peak_vram_bytes'] for r in records),
                   peak_reserved_bytes=max(r['peak_reserved_vram_bytes'] for r in records))
    for source, target in (('evaluation', 'evaluation_seconds'), ('local_serialization', 'local_serialization_seconds'),
                           ('local_hash', 'local_hash_seconds')):
        samples = result['timings'][source]
        if samples and all(t['completed'] for t in samples):
            row[target] = sum(t['seconds'] for t in samples)
    row['checkpoint_publication_mean_seconds'] = mean([p['seconds'] for p in result['publications'] if p['stage'] == 'checkpoint'])
    return row


def summarize(root: Path) -> dict[str, Any]:
    matrix = json.loads((root / 'matrix.json').read_text())
    expected = [f'{architecture}-T{frames}' for architecture, frames in CONDITIONS]
    if [entry['condition'] for entry in matrix['conditions']] != expected:
        raise ValueError('Receipt must enumerate exactly the six authorized conditions')
    rows = []
    reference: dict[str, Any] | None = None
    fairness_keys = ('update', 'pack_ids', 'pack_indices', 'input_frames', 'cumulative_input_frames',
                     'valid_teachers', 'teachers_by_kind', 'original_teachers_by_kind', 'learning_rate')
    compared = []
    reference_name = None
    for entry in matrix['conditions']:
        path = root / entry['condition'] / 'condition.json'
        result = json.loads(path.read_text()) if path.exists() else None
        rows.append(measured_row(entry, result))
        if result is None:
            continue
        if reference is None:
            reference = result
            reference_name = entry['condition']
        else:
            if {k: v for k, v in result['model'].items() if k != 'frames'} != {k: v for k, v in reference['model'].items() if k != 'frames'}:
                raise ValueError('Shared production model configuration changed between conditions')
            if result['runtime'] != reference['runtime'] or result['budget'] != reference['budget']:
                raise ValueError('Runtime or optimizer budget differs between conditions')
            for actual, before in zip(result['records'], reference['records'], strict=False):
                if any(actual[k] != before[k] for k in fairness_keys):
                    raise ValueError('Pack, teacher or learning-rate sequence differs between conditions')
            compared.append(dict(condition=entry['condition'], updates=min(len(result['records']), len(reference['records']))))
    return dict(schema_version=1, scope='Four fixed native-720p clips; performance, not accuracy',
                training_time_scope='decode, MDD, forward/backward, optimizer; excludes evaluation and save',
                warm_step_scope='updates 2 through 10 when measured', rows=rows,
                fairness=dict(status='matched_measured_prefixes', reference=reference_name,
                              compared=compared, fields=list(fairness_keys)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, summarize(args.receipts))
