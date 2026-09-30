"""Cost proposal only: three fixed frames from each existing K=4 prefix."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations
from src.utils.geometry.probabilistic_triangulation import GaussianPrior3D,LaplaceConfig
from src.utils.geometry.probabilistic_triangulation.convergence import convergence_config,triangulate_converged
from src.utils.geometry.triangulation import PinholeCamera


def one(job):
    path,record,settings,frame=job
    torch.set_num_threads(1)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['npz_sha256']:raise ValueError('SHA mismatch')
    with np.load(path) as z:a={k:z[k] for k in z.files}
    cameras=tuple(PinholeCamera(str(i),a['camera_estimated_K'][i],a['camera_estimated_R'][i],a['camera_estimated_t'][i]) for i in range(3))
    distribution=BallGMM2D(*(torch.from_numpy(a[k]) for k in ('gmm2d_means_uv','gmm2d_scale_tril_uv','gmm2d_mixture_logits','gmm2d_presence_logits')))
    obs=frame_observations(distribution,torch.from_numpy(a['source_size_wh'].astype(float)),frame=frame)
    start=time.monotonic()
    result=triangulate_converged(obs,cameras,prior=GaussianPrior3D(np.array(settings['prior_mean_m']),np.diag(settings['prior_covariance_diagonal_m2'])),laplace=LaplaceConfig(125,100),config=convergence_config(settings['boundary_convergence']))
    return {'rally_id':record['rally_id'],'frame':frame,'seconds':time.monotonic()-start,'converged':result.converged,'components':len(result.component_converged),'rounds':result.rounds,'history':result.history,'source_npz_sha256':record['npz_sha256']}


def main():
    parser=argparse.ArgumentParser()
    for key in ('source','plan','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();manifest=json.loads((args.source/'manifest.json').read_text());settings=yaml.safe_load(args.plan.read_text())['degradation']
    jobs=[(args.source/(r['rally_id']+'.npz'),r,settings,frame) for r in manifest['rallies'] for frame in (0,r['frames']//2,r['frames']-1)]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(one,jobs))
    seconds=sum(r['seconds'] for r in rows)/len(rows)
    result={'source_manifest_sha256':hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),'records':rows,'mean_seconds_per_frame':seconds,
        '96_rally_512_frame_4_worker_hours':seconds*96*512/4/3600,
        'caveat':'Nine fixed prefix frames only; not final #935 calibration, no seed/frame quality selection, no generation authorization.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
