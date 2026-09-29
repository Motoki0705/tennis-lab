"""Fixed 12-rally snapshot, without triangulation or success-based selection."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import load_plan
from src.tasks.ball_refiner.refiner_3d.synthetic.observations import load_cameras, make_distribution, perturb_cameras
from src.tasks.ball_refiner.refiner_3d.synthetic.simulation import accepted_rally
from src.tasks.ball_refiner.refiner_3d.synthetic.timebase import resample
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations
from src.utils.configuration import PathResolver, RuntimePathRoots

root = Path(sys.argv[1])
output = Path(__file__).parent
data = Path("/home/kamimura/projects/tennis-lab/data")
roots = RuntimePathRoots(project_root=root, data_root=data, artifact_root=data, output_root=output, checkpoint_root=output, cache_root=output, external_asset_root=output)
plan = load_plan(root / "src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml", PathResolver(roots))
torch.set_num_threads(1)
arrays, records = {}, []
# Run-2 failure frames, fixed before this experiment. Separate diagnostic cohort.
old_failures = {(0,0):79, (0,1):0, (0,2):285, (0,3):27, (1,0):131, (1,2):0, (1,3):130, (2,0):70, (2,1):38, (2,2):14}
for split_index in range(3):
    for index in range(4):
        seed = int(np.random.SeedSequence([plan.values["seed"], split_index, index]).generate_state(1)[0])
        result, _, rng, proposals = accepted_rally(plan, seed=seed, index=index)
        times, positions = resample(result.trajectory_sim.numpy(), native_hz=result.sim_fps, numerator=60000, denominator=1001, max_frames=512)
        source = plan.values["geometry"]["sources"][split_index]
        cameras, sizes = load_cameras(plan.camera_paths[split_index], source["camera_keys"])
        cameras = perturb_cameras(cameras, plan.values["geometry"]["perturbation_per_scene"], rng)
        distribution, masks, meta = make_distribution(positions, cameras, sizes, plan.values["degradation"], rng, rally_index=index)
        frames = [(int(f), "scheduled") for f in np.linspace(0, len(times)-1, 8, dtype=int)]
        frames.append((len(times)//2, "shared_gap"))
        if (split_index,index) in old_failures:
            frames.append((old_failures[(split_index,index)], "run2_failure"))
        for frame, cohort in frames:
            key = f"case_{len(records):03d}"
            obs = frame_observations(distribution, torch.from_numpy(sizes), frame=frame)
            for name, value in dict(means=obs.means_px, covariance=obs.covariance_px2, weights=obs.weights, presence=obs.presence, truth=positions[frame], K=np.stack([c.intrinsic for c in cameras]), R=np.stack([c.rotation for c in cameras]), t=np.stack([c.translation for c in cameras])).items():
                arrays[f"{key}_{name}"] = value
            records.append(dict(key=key, cohort=cohort, split=source["split"], rally=index, seed=seed, frame=frame, gap=bool(masks["occlusion_mask"][:,frame].all()), shared_gap_length=meta["shared_gap_length"], physics_proposals=proposals))
        print(source["split"], index, len(times), flush=True)
np.savez_compressed(output / "cases.npz", **arrays)
(output / "cases.json").write_text(json.dumps(dict(plan=plan.values, hashes=plan.input_hashes, records=records), indent=2)+"\n")
