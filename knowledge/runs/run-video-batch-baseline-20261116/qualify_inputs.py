"""Input screening only; these timings are not sequential baseline results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))


def main() -> None:
    import cv2
    import torch
    from hydra import compose, initialize_config_dir

    import src.utils.hydra  # noqa: F401
    from src.tennis_scene.configuration import PipelineRuntimeConfig
    from src.tennis_scene.generate_dataset.manifest import (
        ClipManifest,
        load_dataset_manifest,
    )
    from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
    from src.utils.inference_recording import InferenceRecorder, recording_context
    from src.utils.video import read_video_frame

    root = REPOSITORY / "outputs/tennis_scene/video_batch_baseline/20261116/qualification"
    source = (REPOSITORY / "data/tennis_multiview/processed/meiji_4cam/dataset").resolve()
    dataset = load_dataset_manifest(source)
    candidates = [f"video_000/clip_{i:03d}" for i in range(3, 11)]
    if any(clip not in dataset.clips for clip in candidates):
        raise ValueError("Screen only completed, registered clips 003..010")
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    with InferenceRecorder(root / "recording.json", devices=("cuda:0",),
                           metadata={"purpose": "input_qualification_only", "candidates": candidates}):
        torch.cuda.set_per_process_memory_fraction(10 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
        with initialize_config_dir(config_dir=str(REPOSITORY / "src/tennis_scene/configs"), version_base="1.3"):
            runtime = PipelineRuntimeConfig.from_config(compose(config_name="pipeline"), bind_inputs=False)
        model = CourtKPModule(runtime.court_kp)
        try:
            for clip_id in candidates:
                clip = ClipManifest.load(source / dataset.clips[clip_id].path)
                row = {"clip_id": clip_id, "num_frames": clip.num_frames, "cameras": [], "all_cameras_accepted": False}
                rows.append(row)
                for camera in ("cam3", "cam1", "cam0", "cam2"):
                    with recording_context(clip_id=clip_id, camera_id=camera):
                        try:
                            packet = read_video_frame(clip.media_path(camera), 0)
                            prediction = model.predict_image(cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB))
                            row["cameras"].append({"camera_id": camera, "status": "accepted", "valid_kp": int(prediction.valid.sum())})
                        except ValueError as exc:
                            row["cameras"].append({"camera_id": camera, "status": "rejected", "error": str(exc)})
                            break
                row["all_cameras_accepted"] = len(row["cameras"]) == 4 and all(r["status"] == "accepted" for r in row["cameras"])
                (root / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                if row["all_cameras_accepted"]:
                    break
        finally:
            model.unload()


if __name__ == "__main__":
    main()
