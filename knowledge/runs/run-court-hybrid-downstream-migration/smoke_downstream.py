import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import build_court_view_record
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.components.blcs import BLCSModule
from src.tennis_scene.pipeline.components.plcs import PLCSModule
from src.tennis_scene.pipeline.court_reference import reference_metadata

parser = argparse.ArgumentParser(
    description="CPU strict-load and visibility propagation audit; controlled synthetic tensors, not accuracy evaluation."
)
parser.add_argument("--project-root", type=Path, default=Path.cwd())
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path.cwd()
with initialize_config_dir(
    config_dir=str(root / "src/tennis_scene/configs"), version_base="1.3"
):
    c = compose(
        config_name="pipeline",
        overrides=[
            "device=cpu",
            f"paths.project_root={args.project_root.resolve()}",
            "court_reference.view_half_turns=[false,false,true]",
            "plcs.save_result=false",
            "blcs.save_result=false",
        ],
    )
runtime = PipelineRuntimeConfig.from_config(c)
contract = runtime.plcs.court_keypoint_contract
views = tuple(
    build_court_view_record(
        camera_id=f"cam{i}", camera_center_court_m=xyz, contract=contract
    )
    for i, xyz in enumerate(((0.0, -18.0, 10.0), (4.0, -18.0, 10.0), (0.0, 18.0, 10.0)))
)
selection = ReferenceViewSelection.create(
    stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(
        runtime.camera_ids
    ),
    selected_views=views,
    reference_camera_id="cam0",
)
document = {}
write_model_artifact_court_keypoint_contract(document, contract)
# Controlled API smoke only: these are deterministic synthetic observations, not accuracy evidence.
rng = np.random.default_rng(42)
points = rng.uniform(0.2, 0.8, (3, 4, 14, 2)).astype(np.float32)
valid = np.ones((3, 4, 14), np.float32)
valid[2, 1, 2] = 0
points[2, 1, 2] = [-0.2, 1.1]
points[:, 2] = 0
valid[:, 2] = 0
records = []
for kind, cls in [("plcs", PLCSModule), ("blcs", BLCSModule)]:
    module = cls(getattr(runtime, kind))
    start = time.monotonic()
    module.load()
    elapsed = time.monotonic() - start
    seen = []
    method = (
        "predict_multiview_observations"
        if kind == "plcs"
        else "predict_multiview_arrays"
    )
    predict = getattr(module._predictor, method)

    def capture(*, _seen=seen, _predict=predict, **kwargs):
        _seen.append(np.array(kwargs["court_vis"], copy=True))
        return _predict(**kwargs)

    setattr(module._predictor, method, capture)
    kwargs = {
        "court_kp": points,
        "court_vis": valid,
        "court_keypoint_document": document,
        "court_reference_provenance": selection.provenance,
        "reference_metadata": reference_metadata(selection, 1, kind),
    }
    start = time.monotonic()
    if kind == "plcs":
        result = module.process(
            human_kp_2d=rng.uniform(0.3, 0.7, (1, 3, 4, 17, 2)).astype(np.float32),
            human_kp_vis=np.ones((1, 3, 4, 17), np.float32),
            track_ids=np.array([0], np.int32),
            **kwargs,
        )
    else:
        result = module.process(
            ball_uv=rng.uniform(0.3, 0.7, (3, 4, 2)).astype(np.float32),
            ball_vis=np.ones((3, 4), np.bool_),
            **kwargs,
        )
    forward_seconds = time.monotonic() - start
    output = result.position if kind == "plcs" else result.ball_3d
    assert np.isfinite(output).all()
    assert len(seen) == 1 and np.array_equal(seen[0], valid)
    rec = {
        "task": kind,
        "checkpoint": str(module.checkpoint),
        "sha256": hashlib.file_digest(
            module.checkpoint.open("rb"), "sha256"
        ).hexdigest(),
        "load_seconds": elapsed,
        "forward_seconds": forward_seconds,
        "model": type(module._predictor.model).__name__,
        "contract": asdict(contract),
        "output_shape": list(output.shape),
        "finite_outputs": True,
        "court_visibility_preserved": True,
        "masked_frame_count": 1,
    }
    records.append(rec)
    print(rec, flush=True)
    del module
    gc.collect()
args.output.write_text(
    json.dumps(
        {
            "scope": "CPU strict load and controlled tensor API smoke; no real-clip accuracy measured",
            "models": records,
        },
        indent=2,
    )
    + "\n"
)
