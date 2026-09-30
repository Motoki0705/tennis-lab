"""Check full fixed-frame coverage, export reconditioned smoke, retain all flags."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset, validate_rally
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json
from src.utils.geometry.probabilistic_triangulation.solver import COMPONENT_METHODS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for key in ('source','audit','plan','output','bundle'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    source=SyntheticDataset(args.source)
    settings=yaml.safe_load(args.plan.read_text())['degradation']['boundary_convergence']
    manifest=copy.deepcopy(source.manifest)
    manifest['status']='running'
    manifest['plan']['degradation']['boundary_convergence']=settings
    manifest['plan']['degradation']['triangulation']='src.utils.geometry.probabilistic_triangulation.convergence.triangulate_converged'
    manifest['reconditioning']={'source':str(args.source),'source_manifest_sha256':sha(args.source/'manifest.json'),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'plan_sha256':sha(args.plan),'note':'Native trajectories, all 2D GMMs, cameras, timestamps, masks and seeds copied without alteration; only integration conditions/diagnostics replaced.'}
    manifest['rallies']=[]
    args.output.mkdir(parents=True)
    rows, diagnostic_rows=[],[]
    for record in source.records:
        rid=record['rally_id'];chunks=[];records=[]
        for path in sorted(args.audit.glob(rid+'-*.npz')):
            with np.load(path) as z:chunks.append({k:z[k] for k in z.files})
            records.extend(json.loads(path.with_suffix('.json').read_text()))
        data={k:np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
        if not np.array_equal(data['frames'],np.arange(record['frames'])) or [r['frame'] for r in records]!=list(range(record['frames'])):
            raise ValueError('Missing/duplicate frames: '+rid)
        arrays=source.load(record)
        for source_key,destination_key in [('means','gmm3d_means_m'),('covariance','gmm3d_covariance_m2'),('weights','gmm3d_weights')]:arrays[destination_key]=data[source_key].astype(np.float32)
        arrays.update(gmm3d_method_codes=data['method_codes'],integration_converged=np.array([r['converged'] for r in records]),
            integration_rounds=np.array([r['rounds'] for r in records],np.uint8),integration_component_converged=data['component_converged'],integration_component_changes=data['component_changes'],
            integration_nll_delta_nat=np.array([r['history'][-1]['nll_delta_nat'] for r in records]))
        new=copy.deepcopy(record)
        new['component_method_labels']=COMPONENT_METHODS
        new['component_method_counts']=dict(Counter(COMPONENT_METHODS[int(v)] for v in data['method_codes'].ravel()))
        new['integration']={'rule':settings,'converged_frames':int(arrays['integration_converged'].sum()),'nonconverged_frames':int((~arrays['integration_converged']).sum()),'history':[r['history'] for r in records]}
        new['reconditioning_source_npz_sha256']=record['npz_sha256']
        new['triangulation_seconds']=sum(r['seconds'] for r in records)
        validate_rally(arrays,new,manifest['plan'])
        path=args.output/(rid+'.npz');np.savez_compressed(path,**arrays)
        new.update(npz_sha256=sha(path),npz_bytes=path.stat().st_size)
        write_json(args.output/(rid+'.json'),new)
        manifest['rallies'].append(new)
        write_json(args.output/'manifest.json',manifest)
        rows.extend(records)
        diagnostic_rows.append({'rally_id':np.full(len(records),rid),'frame':data['frames'],'changes':data['component_changes'],'component_converged':data['component_converged'],'weights':data['weights'],
            'converged':arrays['integration_converged'],'nll_delta':arrays['integration_nll_delta_nat'],'seconds':np.array([r['seconds'] for r in records]),'all_camera_occluded':arrays['occlusion_mask'].all(0),'event_region':arrays['event_region_mask']})
    manifest['status']='complete';write_json(args.output/'manifest.json',manifest)
    # Re-read every exported NPZ through the normal reader, including float32 SPD.
    exported=SyntheticDataset(args.output)
    for record in exported.records:exported.load(record)
    np.savez_compressed(args.bundle/'frame_diagnostics.npz',**{k:np.concatenate([d[k] for d in diagnostic_rows]) for k in diagnostic_rows[0]})
    seconds=np.array([r['seconds'] for r in rows]);converged=sum(r['converged'] for r in rows)
    summary={'frames':len(rows),'converged_frames':converged,'converged_rate':converged/len(rows),'nonconverged_frames':len(rows)-converged,
        'worker_seconds':float(seconds.sum()),'seconds_per_frame':{'mean':float(seconds.mean()),'p50':float(np.median(seconds)),'p95':float(np.quantile(seconds,.95)),'max':float(seconds.max())},
        'by_rally':[{'rally_id':r['rally_id'],'frames':r['frames'],'converged_frames':r['integration']['converged_frames']} for r in exported.records],
        'source_manifest_sha256':sha(args.source/'manifest.json'),'export_manifest_sha256':sha(args.output/'manifest.json'),
        'export_bytes':sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()),'rallies':[{k:r[k] for k in ('rally_id','seed','frames','npz_sha256','npz_bytes')} for r in exported.records],
        'boundary_130':next(r for r in rows if r['rally_id']=='val-00003' and r['frame']==130)}
    diagnostics={k:np.concatenate([d[k] for d in diagnostic_rows]) for k in diagnostic_rows[0]}
    for key in ('all_camera_occluded','event_region'):
        mask=diagnostics[key]
        summary[key]={'frames':int(mask.sum()),'converged_frames':int(diagnostics['converged'][mask].sum())}
    unresolved_mass=(diagnostics['weights']*(~diagnostics['component_converged'])).sum(-1)
    summary['nonconverged_component_mass']={'mean':float(unresolved_mass.mean()),'p50':float(np.median(unresolved_mass)),'p95':float(np.quantile(unresolved_mass,.95)),'frames_above_half':int((unresolved_mass>.5).sum())}
    summary['achieved_delta_quantiles']={key:np.quantile(value,[.5,.95,1]).tolist() for key,value in [('nll_nat',diagnostics['nll_delta']),('evidence_nat',diagnostics['changes'][:,:,0].max(-1)),('mean_m',diagnostics['changes'][:,:,1].max(-1)),('covariance_relative',diagnostics['changes'][:,:,2].max(-1))]}
    write_json(args.bundle/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
