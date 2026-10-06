import hashlib
import json
import runpy
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from src.tasks.ball_refiner.coordinates.evaluation import error_metrics

root = Path('/home/kamimura/projects/tennis-lab/outputs/ball_refiner')
out = root / 'review/gan-only-20261006'
out.mkdir(parents=True, exist_ok=True)
helpers = runpy.run_path('tests/benchmarks/ball_refiner_coordinates_report.py')
rows = []
for dimension in (2, 3):
    run = root / f'train/rope-{dimension}d-gan-only-eventonly/20261006-v4-s42'
    previous = root / f'train/rope-{dimension}d-gan-eventonly/20261006-v3-s42'
    state = json.loads((run / 'state.json').read_text())
    assert state['status'] == 'complete' and state['step'] == 4000
    current_contract = json.loads((run / 'data_contract.json').read_text())
    previous_contract = json.loads((previous / 'data_contract.json').read_text())
    for key in ('manifest_sha256', 'fps', 'train_ids', 'val_ids', 'test_ids', 'evaluation_corruption', 'evaluation_seed'):
        assert current_contract[key] == previous_contract[key], key
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    old_cfg = yaml.safe_load((previous / 'config.yaml').read_text())
    for key in ('steps', 'batch_size', 'learning_rate', 'weight_decay', 'gradient_clip', 'evaluate_every', 'log_every'):
        assert cfg['training'][key] == old_cfg['training'][key], key
    assert cfg['run']['seed'] == old_cfg['run']['seed'] == 42
    assert cfg['corruption'] == old_cfg['corruption']
    for key, value in old_cfg['model'].items():
        assert cfg['model'][key] == value, key
    curve = [json.loads(line) for line in (run / 'learning_curve.jsonl').read_text().splitlines()]
    assert len(curve) == 40 and curve[-1]['step'] == 4000
    for row in curve:
        step = row['step']
        assert row['gan_weight_current'] == min(max((step - 500) / 1000, 0), 1)
        assert abs(row['reconstruction_weight_current'] - (1-min(max((step-2000)/1000,0),1))) < 1e-12
        if step > 3000:
            assert row['reconstruction_weight'] == row['weighted_reconstruction'] == 0
            assert row['total'] == row['weighted_gan']
    with np.load(previous / 'predictions/pred_test.npz', allow_pickle=False) as baseline_npz:
        baseline = {key: baseline_npz[key] for key in baseline_npz.files}
    reference_report = json.loads((previous / 'predictions/diagnostic_metrics.json').read_text())
    for kind, directory in (('best', 'predictions'), ('last', 'predictions_last')):
        checkpoint = run / f'logs/version_0/checkpoints/{kind}.ckpt'
        metadata = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        detail = json.loads((run / directory / 'diagnostic_metrics.json').read_text())
        assert detail['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert detail['checkpoint_step'] == metadata['step']
        assert detail['evaluation_input_sha256'] == reference_report['evaluation_input_sha256']
        if kind == 'last':
            assert metadata['step'] == 4000 and metadata['reconstruction_weight'] == 0 and metadata['gan_weight'] == 1
            assert {int(s['step']) for s in metadata['disc_optimizer']['state'].values()} == {3500}
            assert {int(s['step']) for s in metadata['optimizer']['state'].values()} == {4000}
        with np.load(run / directory / 'pred_test.npz', allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        for key in ('input', 'target', 'missing', 'event', 'rally_id', 'view_id', 'frame_id'):
            np.testing.assert_array_equal(arrays[key], baseline[key])
        repeated = error_metrics(arrays['prediction'], arrays['target'], arrays['missing'], arrays['event'])
        for key in ('all', 'missing', 'observed', 'event'):
            assert repeated[key] == detail[key], key
        row = {'dimensions':dimension, 'kind':kind, 'run':str(run), 'checkpoint':str(checkpoint),
               'step':metadata['step'], 'reconstruction_weight':metadata['reconstruction_weight'], 'gan_weight':metadata['gan_weight'],
               'all_rmse': detail['all']['rmse'], 'missing_rmse':detail['missing']['rmse'], 'observed_rmse':detail['observed']['rmse'],
               'event_rmse':detail['event']['rmse'], 'unit':detail['unit'], 'validation_rmse':detail['validation_rmse'],
               'motion':helpers['motion_metrics'](arrays, current_contract['fps']), 'evaluation_input_sha256':detail['evaluation_input_sha256'],
               'previous':{k:reference_report[k] for k in ('all','missing','observed','event')},
               'previous_motion':helpers['motion_metrics'](baseline, current_contract['fps'])}
        rows.append(row)
    helpers['plot_curves'](run, out / f'{dimension}d-learning-curves.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    steps = [row['step'] for row in curve]
    for key in ('reconstruction_weight_current','gan_weight_current'):
        axes[0].plot(steps, [row[key] for row in curve],label=key)
    for key in ('weighted_reconstruction','weighted_gan','total'):
        axes[1].plot(steps,[row[key] for row in curve],label=key)
    for ax in axes:
        ax.axvline(3000, color='gray', linestyle='--')
        ax.set_xlabel('Generator update'); ax.legend(fontsize=8)
    axes[0].set_title(f'{dimension}D actual coefficients'); axes[1].set_title('Mean weighted training objectives')
    fig.tight_layout(); fig.savefig(out / f'{dimension}d-loss-schedule.png', dpi=140); plt.close(fig)
(out / 'comparison.json').write_text(json.dumps({'rows':rows,'matched_inputs_and_gt':True,'optimizer_counts_verified':True,'all_metrics_recomputed':True,'baseline_changes':['GAN maximum 2 to 1','position coefficient 1 to scheduled 0'],'scope':'synthetic single seed42; no causal isolation or real-video evaluation'}, indent=2)+'\n')
for row in rows:
    print(json.dumps({k:row[k] for k in ('dimensions','kind','step','all_rmse','missing_rmse','observed_rmse','motion')}))
