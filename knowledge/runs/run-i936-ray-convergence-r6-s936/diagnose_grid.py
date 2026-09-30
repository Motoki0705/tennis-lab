"""Inspect voxel leaf resolution on prespecified representative component products."""
import argparse
from itertools import product
from pathlib import Path
import json

import numpy as np
import torch

from audit_smoke import source_rally
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations
from src.utils.geometry.probabilistic_triangulation import CameraGMM, GaussianPrior3D
from src.utils.geometry.probabilistic_triangulation.distributions import camera_subsets
from src.utils.geometry.probabilistic_triangulation.solver import project_with_jacobian
from src.utils.geometry.probabilistic_triangulation.volume import VoxelConfig, _voxel_subset
from src.utils.geometry.probabilistic_triangulation.ray import single_view_moments,RayProposal


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    prior=GaussianPrior3D(np.array([0.,0.,2.]),np.diag([36.,144.,9.]))
    rows=[]
    for rid,frame,component in [('train-00000',0,1),('train-00003',64,17),('val-00003',130,19),('test-00000',100,2)]:
        a,cameras,distribution=source_rally(str(args.source),rid)
        obs=frame_observations(distribution,torch.from_numpy(a['source_size_wh'].astype(float)),frame=frame)
        combos=[(active,np.array(ix,dtype=int)) for active,p in camera_subsets(obs.presence) for ix in product(range(3),repeat=len(active))]
        active,ix=combos[component];selected=tuple(cameras[i] for i in active);means=obs.means_px[active,ix];cov=obs.covariance_px2[active,ix]
        product_obs=CameraGMM(means[:,None],cov[:,None],np.ones((len(active),1)),np.ones(len(active)))
        old=_voxel_subset(product_obs,selected,list(range(len(active))),prior,VoxelConfig(32,9,2048,4),build_leaves=False)
        voxel_mean=old.weights@old.centers
        if len(active)==1:
            results=[single_view_moments(selected[0],means[0],cov[0],prior,n) for n in (12,20,32,48,64)]
        else:
            proposal=RayProposal(selected,means,cov,prior);results=[proposal.integrate(n) for n in (12,20,32,48,64)]
        levels=np.rint(np.log2(old.extent[0]/(32*old.widths[:,0]))).astype(int)
        eigenvalues,eigenvectors=np.linalg.eigh(results[-1][1]);small=eigenvectors[:,0]
        # RMS within-cell uncertainty along the narrowest posterior direction.
        projected_width=np.sqrt(np.sum(old.widths**2*small**2,axis=-1)/12)
        rows.append({'rally_id':rid,'frame':frame,'component':component,'active':active.tolist(),
            'world_box_extent_m':old.extent.tolist(),'initial_cell_width_m':(old.extent/32).tolist(),
            'finest_cell_width_m':(old.extent/(32*2**8)).tolist(),'leaf_mass_by_level':[float(old.weights[levels==i].sum()) for i in range(9)],
            'posterior_principal_sigma_m':np.sqrt(eigenvalues).tolist(),
            'mass_in_cells_with_transverse_rms_above_posterior_sigma':float(old.weights[projected_width>np.sqrt(eigenvalues[0])].sum()),
            'voxel_mean_m':voxel_mean.tolist(),'ray_mean_m':results[-1][0].tolist(),'mean_shift_m':float(np.linalg.norm(voxel_mean-results[-1][0])),
            'ray_consecutive_deltas':[[float(abs(y[2]-x[2])),float(np.linalg.norm(y[0]-x[0])),float(np.linalg.norm(y[1]-x[1])/max(np.linalg.norm(y[1]),np.linalg.norm(x[1])))] for x,y in zip(results[:-1],results[1:])],
            'ray_minus_voxel_log_evidence_nat':results[-1][2]-old.log_evidence})
    args.output.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
