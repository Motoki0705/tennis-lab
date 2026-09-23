"""CPU-only timing of the existing Court geometry path; no scene publication."""
from pathlib import Path
import cProfile
import io
import json
import pstats
import time

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

import src.tasks.court_detection.inference.predictor as predictor_module
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.utils.configuration import PathResolver

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
output = root / "court_cpu_profile"
output.mkdir(exist_ok=False)
runtime = PipelineRuntimeConfig.from_config(OmegaConf.load(root / "pipeline.expanded_pipeline.yaml"))
request = json.loads((root / "pipeline.json").read_text())
predictor = predictor_module.CourtPredictor.load_from_checkpoint(
    Path("/home/kamimura/projects/tennis-lab/ckpt/court_detection/multiscale_depth3/b863df1f01f0.ckpt"),
    device="cpu", resolver=PathResolver(runtime.roots),
    hybrid_config=runtime.court_kp.postprocess,
)
original = predictor_module.estimate_hybrid_homography
captured = {}

def measured_geometry(*args, **kwargs):
    captured["args"], captured["kwargs"] = args, kwargs
    start = time.perf_counter()
    result = original(*args, **kwargs)
    captured["seconds"] = time.perf_counter() - start
    return result

predictor_module.estimate_hybrid_homography = measured_geometry
records = []
for camera, box in [("cam0", (600, 0, 1680, 540)), ("cam1", (0, 270, 1920, 1080)), ("cam2", (0, 270, 1920, 1080))]:
    capture = cv2.VideoCapture(str(Path(request["clip"]) / "media" / f"{camera}.mp4"))
    success, frame = capture.read()
    capture.release()
    assert success
    x0, y0, x1, y1 = box
    image = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    prediction = predictor.predict(image, postprocess="hybrid", heads=("kp", "line"))
    total = time.perf_counter() - start
    profiler = cProfile.Profile()
    profiler.runcall(original, *captured["args"], **captured["kwargs"])
    profiler.dump_stats(str(output / f"{camera}.prof"))
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
    (output / f"{camera}.txt").write_text(stream.getvalue())
    record = {"camera": camera, "frame": 0, "region": box,
              "hybrid_geometry_seconds_unprofiled": captured["seconds"],
              "full_prediction_seconds_on_cpu": total,
              "status": prediction.homography.status}
    records.append(record)
    print(json.dumps(record), flush=True)
    (output / "metrics.json").write_text(json.dumps({
        "device": "cpu", "torch_threads": 1,
        "scope": "3 diagnostic frames; no CUDA work; CPU total is not GPU forward timing; geometry is the same NumPy/SciPy path used in GPU runs",
        "records": records,
    }, indent=2))
