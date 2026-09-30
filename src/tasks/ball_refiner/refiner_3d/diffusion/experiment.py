"""Fail-closed admission of declared single-factor trials and combined candidates."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..condition_audit import audit_manifest
from ..synthetic.configuration import sha256
from ..synthetic.dataset import SyntheticDataset
from ..synthetic.generator import write_json
from .cohort import reference_validation
from .dev_config import load_config
from .dev_training import run_dev_training


def assert_single_factor(candidate: dict[str, Any], control: dict[str, Any], factor: str) -> None:
    normalized = deepcopy(candidate)
    if factor == 'training_rallies':
        if candidate['expected_counts']['train'] <= control['expected_counts']['train']:
            raise ValueError('Training-rally trial must increase train count')
        normalized['expected_counts'] = control['expected_counts']
    elif factor == 'physics_weight':
        if candidate['loss']['physics'] != 10 * control['loss']['physics']:
            raise ValueError('Declared physics trial must change only the weight by 10x')
        normalized['loss']['physics'] = control['loss']['physics']
    elif factor == 'reprojection_weight':
        if candidate['loss']['reprojection'] != 3 * control['loss']['reprojection']:
            raise ValueError('Declared reprojection trial must change only the weight by 3x')
        normalized['loss']['reprojection'] = control['loss']['reprojection']
    else:
        raise ValueError('Unknown declared single factor')
    if normalized != control:
        raise ValueError('Changes beyond the declared single factor')


def assert_combined_candidate(candidate: dict[str, Any], control: dict[str, Any]) -> None:
    """Admit only the measured 64→512 data factor plus 10× physics weight.

    This is a two-factor model candidate, not a single-factor causal diagnosis.
    Validation selection remains fixed separately by reference_validation().
    """
    if (control['expected_counts'] != {'train': 64, 'val': 16, 'test': 16}
            or candidate['expected_counts'] != {'train': 512, 'val': 64, 'test': 64}):
        raise ValueError('Combined candidate requires the declared 64-to-512 counts')
    normalized = deepcopy(candidate)
    normalized['expected_counts'] = control['expected_counts']
    assert_single_factor(normalized, control, 'physics_weight')


def preflight(plan: dict[str, Any]) -> dict[str, Any]:
    """Inspect complete metadata/file hashes before any CUDA allocation or fitting."""
    names = ('dataset', 'control_manifest', 'config', 'generation_plan', 'preflight_output', 'output')
    if any(not Path(plan[key]).is_absolute() for key in names):
        raise ValueError('Experiment paths must be absolute')
    if not {plan[key] for key in ('control_manifest', 'config', 'generation_plan')} <= plan['pinned_files'].keys():
        raise ValueError('Control, config and generation plan must have pinned hashes')
    source = SyntheticDataset(Path(plan['dataset']))  # Running/failed => stop immediately.
    for name, digest in plan['pinned_files'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Pinned experiment input/code changed: ' + name)
    reference_path = Path(plan['control_manifest'])
    reference = json.loads(reference_path.read_text())
    config = asdict(load_config(Path(plan['config'])))
    config['evaluate_updates'] = list(config['evaluate_updates'])
    if plan['factor'] == 'training_rallies_and_physics_weight':
        assert_combined_candidate(config, reference['config'])
    else:
        assert_single_factor(config, reference['config'], plan['factor'])
    generation = json.loads(Path(plan['generation_plan']).read_text())
    if source.manifest['plan'] != generation['expanded_plan']:
        raise ValueError('Generation differs from the preregistered expanded plan')
    audit = audit_manifest(source, expected_counts=config['expected_counts'])
    ids = reference_validation(source.records, reference)
    # Every original train+val rally must be retained byte-for-byte. Additional
    # val/test are not opened by training and cannot alter the primary population.
    records = {r['rally_id']: r for r in source.records}
    for row in reference['read_rallies']:
        if row['rally_id'].startswith('test-') or records.get(row['rally_id'], {}).get('npz_sha256') != row['npz_sha256']:
            raise ValueError('Control train/val rally was changed or removed')
    if plan['factor'] in ('physics_weight', 'reprojection_weight') and audit['manifest_sha256'] != reference['source_manifest_sha256']:
        raise ValueError('Loss-weight trial must use the identical dataset')
    return {'status': 'complete', 'factor': plan['factor'], 'config': config, 'dataset_audit': audit,
            'control_manifest_sha256': sha256(reference_path), 'primary_val_rallies': sorted(ids),
            'primary_val_frames': sum(records[key]['frames'] for key in ids),
            'preserved_control_train_val': len(reference['read_rallies']), 'test_arrays_read': 0}


def run_experiment(plan_path: Path, *, device: str, audit_only: bool = False) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    audit_path = Path(plan['preflight_output'])
    if audit_path.exists() or Path(plan['output']).exists():
        raise FileExistsError('Experiment output already exists; no automatic retry/overwrite')
    audit = preflight(plan)
    audit['plan_sha256'] = sha256(plan_path)
    audit['plan_path'] = str(plan_path)
    write_json(audit_path, audit)
    if audit_only:
        return audit
    result: dict[str, Any] = run_dev_training(
        Path(plan['dataset']), Path(plan['config']), Path(plan['output']),
        device=device, validation_reference=Path(plan['control_manifest']),
    )
    return result
