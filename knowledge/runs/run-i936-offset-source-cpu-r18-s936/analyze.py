"""Fixed saved-input/output diagnostic; oracle translations never become predictions."""
from __future__ import annotations

import json
import resource
import runpy
import time
from collections import defaultdict
from dataclasses import replace
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
from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from src.utils.schema.court_normalization import denormalize_court_position, normalize_court_position

HELPER = Path(__file__).resolve().parent.parent / 'run-i936-reprojection-gap-r17-s936/analyze.py'
projection = runpy.run_path(str(HELPER))['projection']


def describe(values: np.ndarray) -> dict:
    x = np.asarray(values).reshape(-1)
    assert len(x) and np.isfinite(x).all()
    return dict(count=len(x), mean=float(x.mean()), p50=float(np.quantile(x, .5)),
                p95=float(np.quantile(x, .95)), maximum=float(x.max()), rms=float(np.sqrt(np.square(x).mean())))


def main() -> None:
    torch.set_num_threads(1)
    started = time.perf_counter()
    bundle = Path(__file__).resolve().parent
    output = bundle / 'results'
    output.mkdir()
    root = bundle.parent
    directories = {'c': root / 'run-i936-combined512-physics10-r16-s936/collected',
                   'repro3': root / 'run-i936-repro3-512-physics10-r17-s936/collected'}
    manifests = {k: json.loads((p / 'manifest.json').read_text()) for k, p in directories.items()}
    dataset = SyntheticDataset(Path(manifests['c']['dataset']))
    expected = {r['rally_id']: r['npz_sha256'] for r in manifests['c']['read_rallies'] if r['rally_id'].startswith('val-')}
    assert manifests['c']['read_rallies'] == manifests['repro3']['read_rallies']
    records = sorted((r for r in dataset.records if r['rally_id'] in expected), key=lambda r: r['rally_id'])
    assert len(records) == 16
    hashes = {str(HELPER): sha256(HELPER), str(Path(__file__)): sha256(Path(__file__))}
    for path in (Path('src/tasks/ball_refiner/refiner_3d/diffusion/model.py'), Path('src/tasks/ball_refiner/refiner_3d/diffusion/losses.py'),
                 Path('src/tasks/ball_refiner/refiner_3d/diffusion/data.py'), Path('src/utils/schema/court_normalization.py'), Path('src/utils/schema/court.py')):
        hashes[str(path.resolve())] = sha256(path)
    for p in [dataset.directory / 'manifest.json', *(p / 'manifest.json' for p in directories.values())]:
        hashes[str(p)] = sha256(p)
    collected = defaultdict(list)
    paired_errors = defaultdict(list)
    oracle = []
    support_counts = defaultdict(int)
    precision_max = defaultdict(float)
    min_ram = 2**63-1
    window_count = 0
    skipped_windows = []
    for record in records:
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        min_ram = min(min_ram, available)
        if available < 6*1024**3:
            raise MemoryError('Less than 6 GiB RAM available')
        assert time.perf_counter()-started < 300, 'CPU diagnostic exceeded preregistered five minutes'
        rid = record['rally_id']
        assert record['npz_sha256'] == expected[rid]
        arrays = dataset.load(record)
        # Locate via the same manifest schema already validated by dataset.load.
        for p in dataset.directory.rglob(rid + '.npz'):
            hashes[str(p)] = sha256(p)
        truth = arrays['positions_3d_m'].astype(np.float64)
        t = len(truth)
        uv_truth, front_truth, _ = projection(truth, arrays)
        in_image = ~arrays['out_of_frame_mask']
        visible = (~(arrays['occlusion_mask'] | arrays['out_of_frame_mask'])).sum(0)
        predictions = {}
        saved_samples = {}
        for run, directory in directories.items():
            for arm in ('flow', 'regression'):
                path = directory / arm / 'predictions/update-20000' / (rid + '.npz')
                hashes[str(path)] = sha256(path)
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['positions_3d_m'], arrays['positions_3d_m'])
                    samples = saved['samples_m'].astype(np.float64)
                    predictions[f'{run}_{arm}_mean'] = saved['mean_m'].astype(np.float64)[None]
                    predictions[f'{run}_{arm}_samples'] = samples
                    saved_samples[(run, arm)] = predictions[f'{run}_{arm}_mean'], samples
        path = directories['c'] / 'baseline_predictions' / (rid + '.npz')
        hashes[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as saved:
            for method in ('mixture_mean', 'mixture_mean_rts'):
                predictions[method] = saved[method].astype(np.float64)[None]
        mixture = (arrays['gmm3d_means_m'].astype(np.float64) * arrays['gmm3d_weights'].astype(np.float64)[..., None]).sum(1)
        np.testing.assert_allclose(mixture, predictions['mixture_mean'][0], atol=1e-5, rtol=1e-6)
        uv_mixture, front_mixture, _ = projection(mixture, arrays)
        input_error = np.linalg.norm(uv_mixture-uv_truth, axis=-1)
        layers = {'all': np.ones(t, dtype=bool), 'three_visible': visible == 3,
                  'three_visible_input_within5': (visible == 3) & front_mixture.all(0) & (input_error <= 5).all(0)}
        for label, mask in layers.items():
            support_counts[label] += int(mask.sum())
        for name, sample in predictions.items():
            for pred in sample:
                uv, front, _ = projection(pred, arrays)
                delta = uv - uv_truth
                for label, selected in layers.items():
                    mask = in_image & front & selected[None]
                    collected[f'{name}/reprojection/{label}'].append(np.linalg.norm(delta,axis=-1)[mask])
                    collected[f'{name}/error_m/{label}'].append(np.linalg.norm(pred[selected]-truth[selected],axis=-1))
                    paired_errors[f'{name}/{label}'].append(np.concatenate((pred[selected]-truth[selected], mixture[selected]-truth[selected]),axis=-1))
                valid = in_image & front & (visible == 3)[None] & arrays['free_flight_mask'][None]
                support = np.logical_and.reduce([valid[:, j:t-4+j] for j in range(5)])
                trend = sum(delta[:, j:t-4+j] for j in range(5))/5
                collected[f'{name}/five_frame_trend_px'].append(np.linalg.norm(trend,axis=-1)[support])
                collected[f'{name}/five_frame_residual_px'].append(np.linalg.norm(delta[:,2:-2]-trend,axis=-1)[support])
            # Float32 round trip and a single normalized-ULP displacement.
            p32 = torch.tensor(sample[0], dtype=torch.float32)
            norm = normalize_court_position(p32)
            restored = denormalize_court_position(norm)
            ulp = denormalize_court_position(torch.nextafter(norm, torch.full_like(norm, float('inf'))))
            for kind, changed in (('round_trip',restored), ('next_float32',ulp)):
                delta_m = changed.numpy().astype(np.float64)-p32.numpy().astype(np.float64)
                precision_max[kind+'_m'] = max(precision_max[kind+'_m'],float(np.abs(delta_m).max()))
                uv, front, _ = projection(p32.numpy(), arrays)
                changed_uv, changed_front, _ = projection(changed.numpy(), arrays)
                mask = in_image & front & changed_front & (visible==3)[None]
                precision_max[kind+'_three_visible_px'] = max(precision_max[kind+'_three_visible_px'],float(np.linalg.norm(changed_uv-uv,axis=-1)[mask].max()))
        # Oracle constant translation on original, nonoverlapping ownership spans.
        windows = manifests['c']['arms']['flow']['validation'][-1]['windows'][rid]
        for run in manifests:
            for arm in ('flow','regression'):
                assert windows == manifests[run]['arms'][arm]['validation'][-1]['windows'][rid]
        for window in windows:
            start, stop = window['owned_start'], window['owned_stop']
            good = (visible[start:stop]==3) & arrays['free_flight_mask'][start:stop]
            window_count += 1
            if good.sum()<3:
                skipped_windows.append(dict(rally=rid,start=start,stop=stop,support=int(good.sum())))
                continue
            batch = rally_window(arrays,record,start=start,frames=stop-start,allow_nonconverged=True)
            batch = replace(batch,camera_matrices=batch.camera_matrices.double(),means_2d_px=batch.means_2d_px.double(),
                            covariance_2d_px2=batch.covariance_2d_px2.double(),target_positions_m=batch.target_positions_m.double())
            target = torch.tensor(truth[start:stop][None])
            for (run,arm),(means,samples) in saved_samples.items():
                for kind, variants in (('mean',means),('samples',samples)):
                    for index,pred in enumerate(variants):
                        point = torch.tensor(pred[start:stop][None])
                        offset = (point-target)[:,good].mean(1,keepdim=True)
                        alpha = torch.tensor(0.,dtype=torch.float64,requires_grad=True)
                        at = point - alpha*offset
                        losses = dict(x0=(normalize_court_position(at)-normalize_court_position(target)).square().mean(),
                                      reprojection=robust_reprojection(at,batch),physics=gravity_residual(at,batch))
                        shifted = point-offset
                        after = dict(x0=(normalize_court_position(shifted)-normalize_court_position(target)).square().mean(),
                                     reprojection=robust_reprojection(shifted,batch),physics=gravity_residual(shifted,batch))
                        gradients = {k:float(torch.autograd.grad(v,alpha,retain_graph=True)[0]) for k,v in losses.items()}
                        weights = manifests[run]['config']['loss']
                        oracle.append(dict(run=run,arm=arm,kind=kind,sample_index=index,rally=rid,start=start,stop=stop,
                            good_frames=int(good.sum()),translation_m=offset.detach().numpy().ravel().tolist(),
                            before={k:float(v.detach()) for k,v in losses.items()},after={k:float(v) for k,v in after.items()},
                            directional_derivative=gradients,
                            weighted_position_loss_directional_derivative=sum(weights[k]*v for k,v in gradients.items())))
        print(rid,round(time.perf_counter()-started,3),flush=True)
    summaries = {k:describe(np.concatenate(v)) for k,v in collected.items() if sum(len(x) for x in v)}
    correlations = {}
    for key, chunks in paired_errors.items():
        x = np.concatenate(chunks)
        a,b = x[:,:3],x[:,3:]
        norms=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
        eligible=norms>1e-12
        correlations[key]=dict(count=len(x),mean_signed_error_m=a.mean(0).tolist(),
            pearson_per_axis=[float(np.corrcoef(a[:,i],b[:,i])[0,1]) for i in range(3)],
            cosine_mean=float(((a*b).sum(-1)[eligible]/norms[eligible]).mean()))
    oracle_summary = {}
    for run in manifests:
        for arm in ('flow','regression'):
            for kind in ('mean','samples'):
                rows=[r for r in oracle if (r['run'],r['arm'],r['kind'])==(run,arm,kind)]
                oracle_summary[f'{run}_{arm}_{kind}']=dict(windows_samples=len(rows),
                    lowered_reprojection=sum(r['after']['reprojection']<r['before']['reprojection'] for r in rows),
                    negative_reprojection_direction=sum(r['directional_derivative']['reprojection']<0 for r in rows),
                    negative_weighted_position_direction=sum(r['weighted_position_loss_directional_derivative']<0 for r in rows),
                    maximum_physics_change=max(abs(r['after']['physics']-r['before']['physics']) for r in rows),
                    mean_before={k:float(np.mean([r['before'][k] for r in rows])) for k in ('x0','reprojection','physics')},
                    mean_after={k:float(np.mean([r['after'][k] for r in rows])) for k in ('x0','reprojection','physics')},
                    offset_m=describe(np.array([np.linalg.norm(r['translation_m']) for r in rows])))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,label in zip(axes,('three_visible','three_visible_input_within5'),strict=True):
        for name in ('mixture_mean','mixture_mean_rts','c_flow_mean','repro3_flow_mean'):
            values=np.sort(np.concatenate(collected[f'{name}/reprojection/{label}']))
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=name)
        ax.set(xlim=(0,40),ylim=(0,1),xlabel='Reprojection error (px)',ylabel='ECDF',title=label)
        ax.grid(alpha=.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'input-output-ecdf.png',dpi=160);plt.close(fig)
    for name,digest in hashes.items():
        assert sha256(Path(name))==digest,name
    report=dict(frames=sum(r['frames'] for r in records),val_rallies=16,test_arrays_read=0,extra_val_arrays_read=0,
        support_counts=dict(support_counts),statistics=summaries,error_alignment=correlations,
        normalization=dict(scale_xyz_m=list(COURT_COORD_SCALE_XYZ),maximum_errors=dict(precision_max)),
        oracle_summary=oracle_summary,oracle_windows=window_count,oracle_skipped_windows=skipped_windows,
        oracle_definition='GT-derived constant translation per original ownership window using >=3 three-visible free frames; only loss diagnosis, no performance replacement or model change',
        source_hashes=hashes,input_npz_hashes=expected,
        resources=dict(seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,minimum_available_host_bytes=min_ram))
    (output/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'oracle-windows.json').write_text(json.dumps(oracle,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('support_counts','normalization','oracle_summary','resources')},indent=2))


if __name__ == '__main__':
    main()
