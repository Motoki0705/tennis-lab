"""CPU descriptive audit of saved predictions, not an inference/training trial."""
from __future__ import annotations

import gzip
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.tasks.ball_refiner.refiner_3d.diffusion.data import rally_window
from src.tasks.ball_refiner.refiner_3d.diffusion.losses import gravity_residual, robust_reprojection
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.utils.schema.court_normalization import normalize_court_position


def summary(values: np.ndarray) -> dict:
    flat = np.asarray(values).reshape(-1)
    assert len(flat) and np.isfinite(flat).all()
    return dict(count=len(flat), mean=float(flat.mean()),
                p50=float(np.quantile(flat, .5)), p95=float(np.quantile(flat, .95)))


def projection(points: np.ndarray, arrays: dict) -> tuple:
    camera = np.einsum('vij,tj->vti', arrays['camera_true_R'], points.astype(np.float64)) + arrays['camera_true_t'][:, None]
    q = np.einsum('vij,vtj->vti', arrays['camera_true_K'], camera)
    front = camera[..., 2] > 1e-6
    return q[..., :2] / np.where(front, q[..., 2], 1)[..., None], front, camera[..., 2]


def main() -> None:
    torch.set_num_threads(1)
    started = time.perf_counter()
    bundle = Path(__file__).resolve().parent
    results = bundle / 'results'
    results.mkdir()
    root = bundle.parent
    collected = root / 'run-i936-combined512-physics10-r16-s936/collected'
    overlap = root / 'run-i936-combined-overlap-cpu-r17-s936/results'
    manifest = json.loads((collected / 'manifest.json').read_text())
    assert json.loads((overlap / 'manifest.json').read_text())['status'] == 'complete'
    dataset = SyntheticDataset(Path(manifest['dataset']))
    expected = {r['rally_id']: r['npz_sha256'] for r in manifest['read_rallies'] if r['rally_id'].startswith('val-')}
    records = [r for r in dataset.records if r['rally_id'] in expected]
    assert len(records) == 16
    values = defaultdict(list)
    behind = []
    hashes = {}
    losses = defaultdict(list)
    weights = manifest['config']['loss']
    frames = 0
    min_ram = 2**63 - 1
    for record in sorted(records, key=lambda r: r['rally_id']):
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        min_ram = min(available, min_ram)
        if available < 6*1024**3:
            raise MemoryError('Less than 6 GiB available')
        assert record['npz_sha256'] == expected[record['rally_id']]
        arrays = dataset.load(record)
        count = record['frames']
        frames += count
        batch = rally_window(arrays, record, start=0, frames=count, allow_nonconverged=True)
        files = {'baseline': collected / 'baseline_predictions' / (record['rally_id'] + '.npz'),
                 'overlap': overlap / (record['rally_id'] + '.npz')}
        files.update({arm: collected / arm / 'predictions/update-20000' / (record['rally_id'] + '.npz') for arm in ('flow', 'regression')})
        saved = {}
        for name, path in files.items():
            hashes[str(path)] = sha256(path)
            with np.load(path, allow_pickle=False) as data:
                saved[name] = {key: data[key] for key in data.files}
        truth = arrays['positions_3d_m']
        np.testing.assert_array_equal(truth, saved['flow']['positions_3d_m'])
        np.testing.assert_array_equal(truth, saved['overlap']['positions_3d_m'])
        predictions = {'truth': truth, 'RTS': saved['baseline']['mixture_mean_rts'],
            'mixture': saved['baseline']['mixture_mean'],
            'flow': saved['flow']['mean_m'], 'regression': saved['regression']['mean_m'],
            'flow_overlap': saved['overlap']['flow_overlap_samples_m'].mean(0),
            'regression_overlap': saved['overlap']['regression_overlap_samples_m'].mean(0)}
        target_uv, _, _ = projection(truth, arrays)
        in_image = ~arrays['out_of_frame_mask']
        visible = (~(arrays['occlusion_mask'] | arrays['out_of_frame_mask'])).sum(0)
        repro = {name: projection(pred, arrays) for name, pred in predictions.items()}
        rts_error = np.linalg.norm(repro['RTS'][0] - target_uv, axis=-1)
        for name, prediction in predictions.items():
            uv, front, depth = repro[name]
            error = np.linalg.norm(uv - target_uv, axis=-1)
            mask = in_image & front
            for label, subset in (('all', mask), ('three_visible', mask & (visible == 3)[None]),
                ('RTS_under5', mask & repro['RTS'][1] & (rts_error < 5))):
                values[f'{name}/reprojection/{label}'].append(error[subset])
            for cameras in range(4):
                selected = visible == cameras
                values[f'{name}/error_m/cameras{cameras}'].append(np.linalg.norm(prediction[selected]-truth[selected], axis=-1))
            for camera, frame in np.argwhere(in_image & ~front):
                behind.append(dict(method=name, rally=record['rally_id'], camera=int(camera), frame=int(frame),
                    depth_m=float(depth[camera, frame]), visible_cameras=int(visible[frame]),
                    gap=bool(arrays['occlusion_mask'][:, frame].all())))
            # Describe slow error versus residual on a fixed five-frame support,
            # without using the smoothed errors as performance predictions.
            delta = uv-target_uv
            valid = mask & (visible==3)[None] & arrays['free_flight_mask'][None]
            support = np.logical_and.reduce([valid[:, i:count-4+i] for i in range(5)])
            trend = sum(delta[:, i:count-4+i] for i in range(5))/5
            values[f'{name}/five_frame_trend_px'].append(np.linalg.norm(trend, axis=-1)[support])
            values[f'{name}/five_frame_residual_px'].append(np.linalg.norm(delta[:, 2:-2]-trend, axis=-1)[support])
            if name == 'mixture':
                continue
            # Output-coordinate gradients at saved generated positions. These
            # are not parameter gradients or training-flow-time gradients.
            point = torch.tensor(prediction[None], dtype=torch.float32, requires_grad=True)
            terms = {
                'x0': (normalize_court_position(point)-normalize_court_position(batch.target_positions_m)).square().mean(),
                'reprojection': robust_reprojection(point, batch),
                'physics': gravity_residual(point, batch)}
            gradients = {}
            for term, loss in terms.items():
                losses[f'{name}/{term}'].append((float(loss.detach()), count))
                gradient = torch.autograd.grad(loss * weights[term], point, retain_graph=True)[0].detach().numpy()[0] * count
                gradients[term] = gradient
                values[f'{name}/weighted_output_gradient/{term}'].append(np.linalg.norm(gradient, axis=-1))
            a, b = gradients['x0'], gradients['reprojection']
            norms = np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
            eligible = norms > 1e-15
            if eligible.any():
                values[f'{name}/x0_reprojection_gradient_cosine'].append((a*b).sum(-1)[eligible]/norms[eligible])
        covariance = batch.covariance_2d_px2.numpy()[0]
        component_weights = batch.weights_2d.numpy()[0]
        within_rms = np.sqrt((component_weights * np.trace(covariance,axis1=-2,axis2=-1)).sum(-1))
        values['observation/within_component_rms_px/observed'].append(within_rms[in_image & ~arrays['occlusion_mask']])
        values['observation/within_component_rms_px/gap'].append(within_rms[in_image & arrays['occlusion_mask']])
        values['observation/presence/observed'].append(batch.presence_2d.numpy()[0][in_image & ~arrays['occlusion_mask']])
    reduced = {key: summary(np.concatenate(parts)) for key, parts in values.items() if sum(len(p) for p in parts)}
    scalar = {key: sum(v*n for v,n in rows)/sum(n for _,n in rows) for key,rows in losses.items()}
    curves = {}
    for arm in ('flow','regression'):
        with gzip.open(collected/arm/'updates.jsonl.gz','rt') as f:
            rows=[json.loads(line) for line in f]
        curves[arm]={'last_2000_updates_weighted_loss': {k: float(np.mean([r[k]*weights[k] for r in rows[-2000:]])) for k in weights},
                     'fixed_20k_validation_probe': manifest['arms'][arm]['validation'][-1]['loss']}
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, support in zip(axes, ('all','three_visible'), strict=True):
        for name in ('RTS','flow','flow_overlap','regression_overlap'):
            x=np.sort(np.concatenate(values[f'{name}/reprojection/{support}']))
            ax.plot(x,np.arange(1,len(x)+1)/len(x),label=name)
        ax.set(xlim=(0,50),ylim=(0,1),xlabel='Reprojection error (px)',ylabel='ECDF',title=support)
        ax.grid(alpha=.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(results/'reprojection-ecdf.png',dpi=160)
    plt.close(fig)
    report={'frames':frames,'val_rallies':len(records),'test_arrays_read':0,'extra_val_arrays_read':0,
        'source_hashes':hashes,'code_sha256':sha256(Path(__file__)),
        'statistics':reduced,'saved_output_loss_terms':scalar,'loss_weights':weights,'training_loss':curves,
        'behind_incidents':behind,
        'gradient_definition':'configured-weight output-position gradient times rally length; not parameter/training-flow gradients',
        'five_frame_definition':'three-visible free-flight common full five-frame support; analysis only, no replacement predictions',
        'resources':{'elapsed_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'minimum_available_host_bytes':min_ram}}
    (results/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'frames':frames,'resources':report['resources'],'statistics':{k:v for k,v in reduced.items() if 'reprojection/' in k}},indent=2))


if __name__ == '__main__':
    main()
