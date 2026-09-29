"""Validate every rally and prove identity with the preselected comparison inputs."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations

from src.utils.paths import PROJECT_ROOT
root = PROJECT_ROOT
data = Path("/home/kamimura/projects/tennis-lab/data/ball_refiner/synthetic-3d-i936-smoke-r3")
bundle = root / "knowledge/runs/run-i936-triangulation-wide-s936"
torch.set_num_threads(1)
dataset = SyntheticDataset(data)
metadata = json.loads((bundle/"cases.json").read_text())
cases = np.load(bundle/"cases.npz",allow_pickle=False)
checked=0
counts=Counter()
records=[]
for record in dataset.records:
    arrays = dataset.load(record)  # all timestamps/SPDs/subsets/events/gaps/methods/means
    distribution=BallGMM2D(*(torch.from_numpy(arrays[n]) for n in ("gmm2d_means_uv","gmm2d_scale_tril_uv","gmm2d_mixture_logits","gmm2d_presence_logits")))
    for case in metadata["records"]:
        if case["split"] != record["split"] or case["rally"] != int(record["rally_id"].split("-")[1]):
            continue
        key=case["key"]; frame=case["frame"]
        assert case["seed"]==record["seed"]
        assert case["physics_proposals"]==record["physics_proposals"]
        obs=frame_observations(distribution,torch.from_numpy(arrays["source_size_wh"].astype(float)),frame=frame)
        for name, value in dict(means=obs.means_px,covariance=obs.covariance_px2,weights=obs.weights,presence=obs.presence,truth=arrays["positions_3d_m"][frame],K=arrays["camera_estimated_K"],R=arrays["camera_estimated_R"],t=arrays["camera_estimated_t"]).items():
            np.testing.assert_array_equal(value,cases[key+"_"+name])
        checked+=1
    counts.update(record["component_method_counts"])
    records.append({key:record[key] for key in ("rally_id","seed","frames","native_frames","elapsed_seconds","simulation_seconds","triangulation_seconds","npz_bytes","shared_gap_length","components_per_frame","events_hit","events_bounce","float32_zero_weight_components","peak_worker_rss_kib")})
assert len(records)==12 and checked==118
assert {r["shared_gap_length"] for r in records}=={8,16,32,64}
result=dict(status="passed",rallies=records,matched_comparison_frames=checked,component_methods=counts,output_total_bytes=sum(p.stat().st_size for p in data.iterdir() if p.is_file()),failures=dataset.manifest["failures"])
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
