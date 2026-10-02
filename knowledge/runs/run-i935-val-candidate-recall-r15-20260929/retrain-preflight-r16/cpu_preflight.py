"""CPU-only timing/config audit; synthetic heatmaps do not measure accuracy."""
import json
import math
import statistics
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import src.utils.hydra  # registers path resolvers
from src.tasks.ball_detection.configuration import validate_training
from src.tasks.ball_detection.data.store_datamodule import BallStoreDataModule
from src.tasks.ball_detection.training.candidate_recall import ValidationCandidateRecall

WT = Path('/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-10-detector-selection')
MAIN = Path('/home/kamimura/projects/tennis-lab')
OUT = WT / 'outputs/campaign_930/i935/run16'
torch.set_num_threads(4)
assert not torch.cuda.is_initialized()
overrides = [f'paths.project_root={WT}', f'paths.data_root={MAIN}/data',
             f'paths.checkpoint_root={MAIN}/ckpt', f'paths.output_root={MAIN}/outputs',
             f'paths.artifact_root={MAIN}/outputs', f'paths.cache_root={MAIN}/.cache',
             f'paths.external_asset_root={MAIN}/third_party',
             'run.output_dir=ball_detection/train/i935_mixed_ft/s42-r16-20260929']
with initialize_config_dir(config_dir=str(WT / 'src/tasks/ball_detection/configs'), version_base='1.3'):
    config = compose(config_name='train_meiji_mixed', overrides=overrides)
OmegaConf.resolve(config)
validate_training(config)
OmegaConf.save(config, OUT / 'resolved-config.yaml')
old = OmegaConf.to_container(OmegaConf.load(WT / 'knowledge/runs/run-i934-mixed-ft-s42-r6/config.yaml'), resolve=True)
new = OmegaConf.to_container(config, resolve=True)

def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            path = f'{prefix}.{key}' if prefix else key
            if key not in a or key not in b:
                yield {'path': path, 'old': a.get(key), 'new': b.get(key)}
            else:
                yield from differences(a[key], b[key], path)
    elif a != b:
        yield {'path': prefix, 'old': a, 'new': b}

diff = list(differences(old, new))
allowed = {'paths.project_root', 'run.output_dir', 'data.eval_stride',
           'training.checkpoint.monitor', 'training.checkpoint.save_top_k',
           'training.validation_candidates'}
assert {d['path'] for d in diff} <= allowed, diff
(OUT / 'config-diff-vs-r6.json').write_text(json.dumps(diff, indent=2) + '\n')
data = BallStoreDataModule(config)
data.pin_memory = False  # timing only, no CUDA context
data.setup('validate')
dataset = data.val_dataset
assert dataset is not None
by_source = {}
for i in range(len(dataset)):
    by_source.setdefault(dataset.source_of(i), []).append(i)
indices = []
for values in by_source.values():
    indices.extend(values[round(j * (len(values) - 1) / 63)] for j in range(64))
loader = DataLoader(Subset(dataset, indices), batch_size=4, num_workers=4, pin_memory=False)
started = time.perf_counter()
load_times = []
previous = started
first_batch = None
for batch in loader:
    now = time.perf_counter()
    load_times.append(now - previous)
    if first_batch is None:
        first_batch = batch
    previous = now
load_total = time.perf_counter() - started
del loader
assert first_batch is not None
recall = ValidationCandidateRecall(config.training.validation_candidates)
torch.manual_seed(42)
heatmaps = torch.rand(4, 8, 72, 128)
timings = []
for iteration in range(65):
    recall.reset()
    started = time.perf_counter()
    recall.update(heatmaps, first_batch)
    if iteration:
        timings.append(time.perf_counter() - started)
assert not torch.cuda.is_initialized()
stats = json.loads((WT / 'knowledge/runs/run-i934-mixed-ft-s42-r6/training_result.json').read_text())['scalars']
train = stats['train/loss']
old_val = []
for value in stats['val/f1']:
    points = [v for v in train if v['step'] // 1920 == value['step'] // 1920]
    slope = (points[-1]['wall_time'] - points[-2]['wall_time']) / (points[-1]['step'] - points[-2]['step'])
    train_end = points[-1]['wall_time'] + (value['step'] - points[-1]['step']) * slope
    old_val.append(value['wall_time'] - train_end)
old_total = stats['val/f1'][-1]['wall_time'] - stats['hp_metric'][0]['wall_time']
report = {'cuda_initialized': torch.cuda.is_initialized(), 'num_cpu_workers': 4,
          'recipe_overrides': overrides, 'config_diff': diff,
          'validation_windows': len(dataset), 'validation_batches': math.ceil(len(dataset) / 4),
          'windows_by_source': {k: len(v) for k, v in by_source.items()},
          'sampled_windows_per_source': 64, 'sampled_batches': len(load_times),
          'sampled_loading_total_seconds_including_worker_start_stop': load_total,
          'loading_seconds_per_batch_mean_excluding_first': statistics.mean(load_times[1:]),
          'loading_seconds_per_batch_p90_excluding_first': sorted(load_times[1:])[int(.9 * len(load_times[1:]))],
          'synthetic_candidate_native_shape': list(heatmaps.shape),
          'candidate_cpu_seconds_per_batch_mean': statistics.mean(timings),
          'candidate_cpu_seconds_per_batch_p90': sorted(timings)[int(.9 * len(timings))],
          'old_validation_windows': 4633, 'old_total_seconds': old_total,
          'old_validation_seconds_per_epoch_estimated': old_val,
          'old_nonvalidation_seconds_estimated': old_total - sum(old_val),
          'notes': ['No model forward or GPU; random heatmaps measure processing cost only.',
                    'Old validation wall time subtracts an interpolated final training timestamp.',
                    'Data loading and candidate timing are sequential, not five simultaneous CPU-heavy processes.',
                    'Probe disables pin_memory only; resolved training config retains r6 pin_memory=true.']}
(OUT / 'cpu-preflight.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('notes', 'recipe_overrides')}, indent=2))
