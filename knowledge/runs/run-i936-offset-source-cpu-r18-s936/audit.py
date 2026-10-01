"""Verify source parity, every oracle reprojection derivative, and all sample precision."""
from dataclasses import replace
import json
from math import isclose
from pathlib import Path
import runpy

import numpy as np
import torch

from src.tasks.ball_refiner.refiner_3d.diffusion.data import rally_window
from src.tasks.ball_refiner.refiner_3d.diffusion.losses import robust_reprojection
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.utils.schema.court_normalization import denormalize_court_position, normalize_court_position


def main() -> None:
    torch.set_num_threads(1)
    b = Path(__file__).resolve().parent
    j = json.loads((b/'results/analysis.json').read_text())
    rows = json.loads((b/'results/oracle-windows.json').read_text())
    projection = runpy.run_path(str(b/'analyze.py'))['projection']
    directories = {'c': b.parent/'run-i936-combined512-physics10-r16-s936/collected',
                   'repro3': b.parent/'run-i936-repro3-512-physics10-r17-s936/collected'}
    manifests = {k:json.loads((d/'manifest.json').read_text()) for k,d in directories.items()}
    n = 0
    for run,d in directories.items():
        ref = json.loads((d/'comparison.json').read_text())['methods']
        for arm in ('flow','regression'):
            for kind in ('mean','samples'):
                name = f'{run}_{arm}_{kind}'
                m = ref[f'{arm}_20000_{kind}']['metrics']
                for q in ('mean','p50','p95','count'):
                    assert isclose(j['statistics'][name+'/reprojection/all'][q],m['reprojection_px_all'][q],rel_tol=1e-8,abs_tol=1e-8)
                    n += 1
                assert isclose(j['statistics'][name+'/error_m/all']['rms'],m['rmse_m_overall']['value'],rel_tol=1e-8,abs_tol=1e-8)
                n += 1
    assert sum('/data/ball_refiner/' in p and p.endswith('.npz') for p in j['source_hashes']) == 16
    for path,digest in j['source_hashes'].items():
        assert sha256(Path(path)) == digest
    dataset = SyntheticDataset(Path(manifests['c']['dataset']))
    records = {r['rally_id']:r for r in dataset.records if r['rally_id'] in j['input_npz_hashes']}
    checked = 0
    max_difference = 0.
    precision = {'round_trip_m':0., 'round_trip_three_visible_px':0.,'next_float32_m':0.,'next_float32_three_visible_px':0.}
    precision_paths = 0
    for rid, record in records.items():
        arrays = dataset.load(record)
        for suffix in ('K','R','t'):
            np.testing.assert_array_equal(arrays['camera_true_'+suffix],arrays['camera_estimated_'+suffix])
        good = (~(arrays['occlusion_mask'] | arrays['out_of_frame_mask'])).sum(0)==3
        paths = {}
        for run,d in directories.items():
            for arm in ('flow','regression'):
                with np.load(d/arm/'predictions/update-20000'/f'{rid}.npz',allow_pickle=False) as saved:
                    paths[(run,arm,'mean')]=saved['mean_m'][None].astype(np.float64)
                    paths[(run,arm,'samples')]=saved['samples_m'].astype(np.float64)
        for variants in paths.values():
            for pred in variants:
                point = torch.tensor(pred,dtype=torch.float32)
                norm = normalize_court_position(point)
                uv,front,_ = projection(point.numpy(),arrays)
                for label,changed in [('round_trip',denormalize_court_position(norm)),('next_float32',denormalize_court_position(torch.nextafter(norm,torch.full_like(norm,float('inf')))))]:
                    precision[label+'_m']=max(precision[label+'_m'],float((changed-point).abs().max()))
                    altered,valid,_=projection(changed.numpy(),arrays)
                    mask=front&valid&good[None]&~arrays['out_of_frame_mask']
                    precision[label+'_three_visible_px']=max(precision[label+'_three_visible_px'],float(np.linalg.norm(altered-uv,axis=-1)[mask].max()))
                precision_paths+=1
        for row in (r for r in rows if r['rally']==rid):
            start,stop=row['start'],row['stop']
            batch=rally_window(arrays,record,start=start,frames=stop-start,allow_nonconverged=True)
            batch=replace(batch,camera_matrices=batch.camera_matrices.double(),means_2d_px=batch.means_2d_px.double(),covariance_2d_px2=batch.covariance_2d_px2.double())
            point=torch.tensor(paths[(row['run'],row['arm'],row['kind'])][row['sample_index'],start:stop][None])
            offset=torch.tensor(row['translation_m']).reshape(1,1,3).double()
            # Preserve the JSON float64 offset; torch.tensor defaults to float32.
            offset=torch.tensor(row['translation_m'],dtype=torch.float64).reshape(1,1,3)
            step=1e-4
            finite=float((robust_reprojection(point-step*offset,batch)-robust_reprojection(point+step*offset,batch))/(2*step))
            exact=row['directional_derivative']['reprojection']
            assert isclose(finite,exact,rel_tol=1e-4,abs_tol=1e-6),(rid,finite,exact)
            max_difference=max(max_difference,abs(finite-exact))
            checked+=1
    result=dict(original_metric_checks=n,source_hashes_verified=len(j['source_hashes']),camera_estimated_true_parity_rallies=len(records),
                oracle_finite_difference_checks=checked,maximum_derivative_difference=max_difference,
                precision_mean_and_sample_paths=precision_paths,all_sample_precision=precision,
                maximum_translation_physics_change=max(r['maximum_physics_change'] for r in j['oracle_summary'].values()),
                code_sha256=sha256(Path(__file__)),test_arrays_read=0,extra_val_arrays_read=0)
    (b/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
