"""Reproduce the non-convergence of voxel budget sensitivity on case_078."""
import json, time, sys
from pathlib import Path
import numpy as np
from src.utils.geometry.probabilistic_triangulation import CameraGMM, GaussianPrior3D, LaplaceConfig
from src.utils.geometry.probabilistic_triangulation.solver import HybridConfig, triangulate_hybrid
from src.utils.geometry.probabilistic_triangulation.volume import VoxelConfig
from src.utils.geometry.triangulation import PinholeCamera
root=Path(__file__).parent
case=np.load(root/"cases.npz"); key="case_078"
obs=CameraGMM(*(case[key+"_"+n] for n in ("means","covariance","weights","presence")))
cameras=tuple(PinholeCamera(str(i),case[key+"_K"][i],case[key+"_R"][i],case[key+"_t"][i]) for i in range(3))
prior=GaussianPrior3D(np.array([0.,0.,2.]),np.diag([36.,144.,9.]))
records=[]
for initial,levels,refine in [(24,8,1024),(32,9,2048)]:
    start=time.perf_counter()
    result=triangulate_hybrid(obs,cameras,prior=prior,config=HybridConfig(LaplaceConfig(64,100),VoxelConfig(initial,levels,refine,4.))).distribution
    records.append(dict(initial_cells=initial,levels=levels,refine_cells=refine,nll=-float(result.log_prob(case[key+"_truth"])),seconds=time.perf_counter()-start,weights=result.weights.tolist()))
Path(sys.argv[1]).write_text(json.dumps(records,indent=2)+"\n")
