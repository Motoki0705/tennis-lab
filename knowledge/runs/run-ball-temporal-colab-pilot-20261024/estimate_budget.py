"""Explicit future-cost scenarios; projections never enter the measured table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from session_guard import write
from summarize import summarize

TEST_INTERVALS = Path('knowledge/runs/run-ball-all-frame-protocol-20261022/common-intervals-test.jsonl')


def completed_measurements(receipts: Path) -> dict[str, Any]:
    table = summarize(receipts)
    if any(row['status'] != 'completed' for row in table['rows']):
        raise ValueError('A full-matrix cost scenario requires all six measured conditions; do not fill missing data')
    return {row['condition']: row for row in table['rows']}


def test_workloads() -> dict[str, int]:
    # Same stride and tail rule as the pinned protocol; these are CPU counts,
    # not measured test inference times or a replacement evaluation manifest.
    spans = [span for line in TEST_INTERVALS.read_text().splitlines() for span in json.loads(line)[1]]
    return {str(t): sum(((b-a-t) // (t//2) + 1 + int((b-a-t) % (t//2) != 0)) * t
                        for a, b in spans) for t in (32, 64, 128)}


def scenario(
    measurements: dict[str, Any], raw: dict[str, Any], preparation: dict[str, Any], *,
    name: str, pre: int, post: int, pre_warmup: int, pre_evaluation: int, post_evaluation: int,
    repeat_pretraining: bool, train_setup_hours: float, rate: float,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        if seed == 42 or repeat_pretraining:
            runs.extend(dict(architecture=a, frames=32, seed=seed, phase='pretrain', updates=pre,
                             evaluations=math.ceil(pre / pre_evaluation)) for a in ('dpt', 'hybrid'))
        lengths = (32, 64, 128) if seed == 42 else (32, 128)
        runs.extend(dict(architecture=a, frames=t, seed=seed, phase='finetune', updates=post,
                         evaluations=math.ceil(post / post_evaluation)) for t in lengths for a in ('dpt', 'hybrid'))
    test_counts = test_workloads()
    training = validation = test = persistence = 0.
    for run in runs:
        key = f"{run['architecture']}-T{run['frames']}"
        measured, receipt = measurements[key], raw[key]
        t = str(run['frames'])
        diagnostic_input = preparation['diagnostic_evaluation_counts'][t]['windows'] * run['frames']
        seconds_per_evaluation_frame = measured['evaluation_seconds'] / diagnostic_input
        training += run['updates'] * measured['warm_step_mean_seconds']
        training += max(0., measured['first_step_seconds'] - measured['warm_step_mean_seconds'])
        validation += run['evaluations'] * preparation['full_validation_workloads'][t]['input_frames'] * seconds_per_evaluation_frame
        if run['phase'] == 'finetune':
            test += test_counts[t] * seconds_per_evaluation_frame
        checkpoint_count = math.ceil(run['updates'] / 256)
        # Twofold publication allowance for large manifests/transfer variance;
        # it is an assumption, not another measured save time.
        local_save = (measured['local_serialization_seconds'] + measured['local_hash_seconds']) / 2
        persistence += checkpoint_count * (2 * measured['checkpoint_publication_mean_seconds'] + local_save)
        persistence += sum(p['seconds'] for p in receipt['publications'] if p['stage'] != 'checkpoint')
    # Future session segmentation is a chosen policy, not a Colab lifetime promise.
    session_hours, input_check_hours, initial_manifest_hours, test_setup_hours = 8., 1/6, 1.5, .35
    if not 0 < train_setup_hours < session_hours or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Setup time and observed rate must be finite positive planning inputs')
    trainval = (training + validation + persistence) / 3600 + initial_manifest_hours + len(runs) * input_check_hours
    train_sessions = 1
    for _ in range(20):
        updated = math.ceil((trainval + (train_sessions-1) * input_check_hours) / (session_hours - train_setup_hours))
        if updated == train_sessions:
            break
        train_sessions = updated
    else:
        raise ValueError('Session-count estimate did not converge')
    test_hours = test / 3600
    test_sessions = math.ceil(test_hours / (session_hours - test_setup_hours))
    raw_hours = (trainval + (train_sessions-1) * input_check_hours + train_sessions * train_setup_hours
                 + test_hours + test_sessions * test_setup_hours)
    return dict(
        name=name, status='proposal_not_approved', Upre=pre, Upost=post,
        pretrain_warmup_updates=pre_warmup, posttrain_warmup_updates=64,
        pretrain_evaluation_interval=pre_evaluation, posttrain_evaluation_interval=post_evaluation,
        checkpoint_interval=256, input_frames_per_update=128, effective_windows=dict(T32=4, T64=2, T128=1),
        additional_seeds=[43, 44], additional_seed_lengths=[32, 128],
        additional_seed_pretraining='repeat' if repeat_pretraining else 'reuse_seed42_parent',
        learning_rate=.0001, minimum_lr_ratio=.1, weight_decay=.01, gradient_clip=1.,
        schedule='linear_warmup_cosine', augmentation='none', run_count=len(runs), runs=runs,
        projected_training_hours=training/3600, projected_validation_hours=validation/3600,
        projected_test_hours=test_hours, projected_save_hours=persistence/3600,
        planned_trainval_sessions=train_sessions, planned_test_sessions=test_sessions,
        projected_total_allocation_hours=raw_hours, planning_allocation_hours_with_25_percent_allowance=raw_hours*1.25,
        projected_compute_units=raw_hours*rate, planning_compute_units_with_allowance=raw_hours*1.25*rate,
        proposed_gpu_hours_ceiling=200, proposed_compute_unit_ceiling=310,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('receipts', 'preparation', 'environment', 'data', 'doctor', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    measured = completed_measurements(args.receipts)
    raw = {key: json.loads((args.receipts / key / 'condition.json').read_text()) for key in measured}
    preparation = json.loads(args.preparation.read_text())
    environment = json.loads(args.environment.read_text())
    data = json.loads(args.data.read_text())
    usage = json.loads(args.doctor.read_text())['usage']
    rate = float(usage.split('Usage rate: ')[1].split('/hr')[0])
    setup = (data['total_seconds'] + environment['total_seconds'] + preparation['restoration']['seconds'] + 60) / 3600
    proposals = [
        scenario(measured, raw, preparation, name='A_longer_parent_shared_extra_seeds', pre=6144, post=1536, pre_warmup=256,
                 pre_evaluation=2048, post_evaluation=1536, repeat_pretraining=False, train_setup_hours=setup, rate=rate),
        scenario(measured, raw, preparation, name='B_shorter_budget_fresh_extra_parents', pre=3072, post=1024, pre_warmup=128,
                 pre_evaluation=1536, post_evaluation=1024, repeat_pretraining=True, train_setup_hours=setup, rate=rate),
    ]
    write(args.output, dict(
        schema_version=1, kind='unapproved_cost_projection', observed_rate_cu_per_hour=rate,
        trainval_vm_setup_hours_with_dispatch_allowance=setup, test_workloads_input_frames=test_workloads(),
        test_workload_source=TEST_INTERVALS.as_posix(), test_workload_source_sha256=hashlib.sha256(TEST_INTERVALS.read_bytes()).hexdigest(),
        assumptions=dict(planned_session_hours=8, no_guaranteed_vm_lifetime=True,
                         extra_dispatch_allowance_seconds_per_vm=60,
                         test_only_vm_setup_hours=.35, full_input_verification_hours_per_run_or_extra_vm=1/6,
                         first_full_manifest_preparation_allowance_hours=1.5, publication_time_multiplier=2,
                         overall_time_allowance_ratio=.25, all_sessions_sequential=True),
        limitations=[
            'Validation/test timings are linear input-frame projections from 162 unique diagnostic frames, not full-data measurements.',
            'Large prediction writing, full-data calibration, cold random I/O and runtime interruptions can exceed these allowances.',
            'Test-only dataset lineage and calibration integration must be fixed before scientific training; current CLI rejects that path.',
            'The inspected test-distribution Drive path was absent. Initial test upload and integration work are unmeasured calendar-time additions outside these GPU projections.',
            'Shared extra seeds assess fine-tuning variation only; ten pilot updates do not establish convergence or an optimal LR.',
            'The proposed ceilings stop work; they do not authorize purchases, budget extensions or changing fixed scientific updates.',
            'Convergence-extension experiments are not included in these scenarios and require a separate fixed budget.',
        ], scenarios=proposals))


if __name__ == '__main__':
    main()
