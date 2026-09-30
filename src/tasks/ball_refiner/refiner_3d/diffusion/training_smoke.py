"""CPU overfit/plumbing on verified saved rallies, with all four x0-flow losses."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
import yaml

from src.tasks.ball_refiner.refiner_3d.diffusion.data import rally_window
from src.tasks.ball_refiner.refiner_3d.diffusion.flow import (
    sample_trajectories,
    training_objective,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.losses import LossConfig, TrainingBatch
from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    MixtureCondition,
    ModelConfig,
    TrajectoryDenoiser,
)
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json


def _collate(batches: list[TrainingBatch]) -> TrainingBatch:
    condition = MixtureCondition(**{field.name: torch.cat([getattr(b.condition, field.name) for b in batches]) for field in fields(MixtureCondition)})
    return TrainingBatch(condition=condition, **{field.name: torch.cat([getattr(b, field.name) for b in batches]) for field in fields(TrainingBatch) if field.name != 'condition'})


def run_training_smoke(dataset: Path, config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    raw = yaml.safe_load(config_path.read_text())
    required = {'device','seed','updates','frames','learning_rate','maximum_seconds','allow_nonconverged','model','loss'}
    if set(raw) != required or raw['device'] != 'cpu' or type(raw['allow_nonconverged']) is not bool:
        raise ValueError('Require complete, explicit CPU overfit configuration')
    for key in ('seed','updates','frames','maximum_seconds'):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f'Invalid {key}')
    if raw['updates'] > 1000 or raw['maximum_seconds'] > 1100 or raw['frames'] < 3:
        raise ValueError('CPU smoke budget exceeded')
    if not math.isfinite(raw['learning_rate']) or raw['learning_rate'] <= 0:
        raise ValueError('Invalid learning rate')
    torch.set_num_threads(1)
    torch.manual_seed(raw['seed'])
    started = time.perf_counter()
    source = SyntheticDataset(dataset)
    if source.manifest['counts'] != {'train':4,'val':4,'test':4}:
        raise ValueError('This diagnostic requires the fixed 12-rally smoke')
    model_config, loss_config = ModelConfig(**raw['model']), LossConfig(**raw['loss'])
    model = TrajectoryDenoiser(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=raw['learning_rate'])
    generator = torch.Generator().manual_seed(raw['seed']+1)
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {'status':'running','diagnostic_only':True,'device':'cpu','config':raw,
        'source_manifest_sha256':sha256(dataset/'manifest.json'),'config_sha256':sha256(config_path),
        'dataset':str(dataset),'parameters':sum(p.numel() for p in model.parameters()),
        'validation_scope':'all 12 rallies plumbing only; tiny overfit uses first two sorted train prefixes; no quality selection or holdout claim'}
    write_json(output/'manifest.json',manifest)
    try:
        batches: list[TrainingBatch] = []
        plumbing: list[dict[str, Any]] = []
        identities: list[dict[str, Any]] = []
        for record in sorted(source.records, key=lambda r:r['rally_id']):
            arrays = source.load(record)
            windows = []
            starts = [min(s, record['frames']-3) for s in range(0,record['frames'],raw['frames'])]
            for start in starts:
                batch = rally_window(arrays, record, start=start, frames=raw['frames'], allow_nonconverged=raw['allow_nonconverged'])
                with torch.no_grad():
                    loss, terms = training_objective(model,batch,loss_config,generator,objective='flow')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite plumbing loss')
                windows.append({'start':start,'real_frames':int((~batch.condition.padding_mask).sum()),
                    'means_shape':list(batch.condition.means_m.shape),'means_2d_shape':list(batch.means_2d_px.shape),
                    'loss':loss.item(),**{k:v.item() for k,v in terms.items()}})
                if record['split']=='train' and start==0 and len(batches)<2:
                    batches.append(batch)
                    identities.append({'rally_id':record['rally_id'],'start':0,'frames':raw['frames'],'nonconverged_frames':int((~arrays['integration_converged'][:raw['frames']]).sum())})
            plumbing.append({'rally_id':record['rally_id'],'frames':record['frames'],
                'nonconverged_frames':int((~arrays['integration_converged']).sum()),'npz_sha256':record['npz_sha256'],'windows':windows})
        write_json(output/'plumbing.json',plumbing)
        tiny = _collate(batches)
        manifest['overfit_windows'] = identities
        manifest['nonconverged_overfit_frames'] = sum(r['nonconverged_frames'] for r in identities)

        def evaluate() -> dict[str, float]:
            model.eval()
            with torch.no_grad():
                loss, terms = training_objective(model,tiny,loss_config,torch.Generator().manual_seed(raw['seed']+2),objective='flow')
            model.train()
            return {'loss':loss.item(),**{k:v.item() for k,v in terms.items()}}

        def sample_rmse() -> float:
            samples = sample_trajectories(model,tiny.condition,samples=4,steps=8,generator=torch.Generator().manual_seed(raw['seed']+3))
            error = (samples.mean_m-tiny.target_positions_m).square().sum(-1)
            return float(error[~tiny.condition.padding_mask].mean().sqrt())

        manifest['initial'] = evaluate()
        manifest['initial_sample_rmse_m'] = sample_rmse()
        write_json(output/'manifest.json',manifest)
        with (output/'updates.jsonl').open('w') as log:
            for update in range(1,raw['updates']+1):
                if time.perf_counter()-started > raw['maximum_seconds']:
                    raise TimeoutError('CPU overfit exceeded its wall-clock budget')
                optimizer.zero_grad(set_to_none=True)
                loss, terms = training_objective(model,tiny,loss_config,generator,objective='flow')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                row: dict[str, Any] = {'update':update,'loss':loss.item(),'gradient_norm':norm.item(),**{k:v.item() for k,v in terms.items()}}
                if update==1 or update%20==0 or update==raw['updates']:
                    row['fixed_probe'] = evaluate()
                log.write(json.dumps(row,allow_nan=False)+'\n')
                log.flush()
        checkpoint = output/'diagnostic-only.pt'
        torch.save({'diagnostic_only':True,'model':asdict(model_config),'state_dict':model.state_dict(),'config':raw,'source_manifest_sha256':manifest['source_manifest_sha256']},checkpoint)
        manifest.update(status='complete',final=evaluate(),final_sample_rmse_m=sample_rmse(),
            elapsed_seconds=time.perf_counter()-started,checkpoint_sha256=sha256(checkpoint),checkpoint_bytes=checkpoint.stat().st_size)
    except Exception as exc:
        manifest.update(status='failed',error=str(exc),elapsed_seconds=time.perf_counter()-started)
        write_json(output/'manifest.json',manifest)
        raise
    write_json(output/'manifest.json',manifest)
    return manifest
