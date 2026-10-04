"""Replay every product of the ten run-2 failure frames, recording raw fits."""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import legacy_solver as legacy

from src.utils.geometry.probabilistic_triangulation import GaussianPrior3D

root = Path(__file__).parent
metadata = json.loads((root/"cases.json").read_text())
cases = np.load(root/"cases.npz",allow_pickle=False)
prior=GaussianPrior3D(np.array([0.,0.,2.]),np.diag([36.,144.,9.]))
original=legacy.least_squares
current, records = {}, []
def capture(*args, **kwargs):
    fit=original(*args, **kwargs)
    matrices=current["matrices"]
    depths=matrices[:,2,:3] @ fit.x + matrices[:,2,3]
    if not fit.success or (depths<=0).any():
        records.append(dict(case=current["case"], active=current["active"], combination=current["combination"], success=bool(fit.success), message=fit.message, nfev=int(fit.nfev), cost=float(fit.cost), optimality=float(fit.optimality), point=fit.x.tolist(), depths=depths.tolist()))
    return fit
legacy.least_squares=capture
for row in metadata["records"]:
    if row["cohort"]!="run2_failure":
        continue
    key=row["key"]
    matrices=np.einsum("vij,vjk->vik", cases[key+"_K"],np.concatenate((cases[key+"_R"],cases[key+"_t"][...,None]),axis=-1))
    for active,_ in legacy.camera_subsets(cases[key+"_presence"]):
        for combination in product(range(3), repeat=len(active)):
            index=np.array(combination,dtype=int)
            current=dict(case=row,active=active.tolist(),combination=list(combination),matrices=matrices[active])
            try:
                legacy.fit_component(matrices[active],cases[key+"_means"][active,index],cases[key+"_covariance"][active,index],prior,max_nfev=100)
            except RuntimeError:
                pass  # Recorded raw failure above; no input/product is removed.
Path(sys.argv[1]).write_text(json.dumps(records,indent=2,allow_nan=False)+"\n")
print(json.dumps(records,indent=2,allow_nan=False))
