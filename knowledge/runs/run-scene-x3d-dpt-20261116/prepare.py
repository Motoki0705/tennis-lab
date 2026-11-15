"""Verify the original snapshot, reproduce validation calibration and export scene metadata.

Run from the checkout root with .venv/bin/python. No model execution or GPU use.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.tasks.ball_detection.evaluation.temporal_heatmap_metrics import calibrate, frame_metrics
from src.tasks.ball_detection.inference.x3d_dpt import scene_dpt_input_contract

FULL_SHA = "8a10a5a856703946d36df4a9490a62f68e05522b96bde0730f648f3e7c9b053e"
WEIGHTS_SHA = "8b06768307e4451396e91f4baa5023b8ef9aa97d053b7088dd17b61a81c27c2c"
FRAMES_SHA = "c121d00dde82506c5dcfe25e812b7c706ca6a91a4d3e0db71434ce8137875307"
MANIFEST_SHA = "fb0ef368e1535bccb5ca1997b39c2e32871fd1f72cb9bb077d4ffe10ac41653d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = {"full": args.snapshot / "checkpoint-update-00010000.pt",
             "weights": args.snapshot / "weights-update-00010000.pt",
             "frames": args.snapshot / "validation-update-00010000.frames.jsonl",
             "manifest": args.snapshot / "evaluation_manifest.json"}
    expected = dict(full=FULL_SHA, weights=WEIGHTS_SHA, frames=FRAMES_SHA, manifest=MANIFEST_SHA)
    actual = {key: sha(path) for key, path in paths.items()}
    if actual != expected:
        raise ValueError(f"Snapshot digest mismatch: {actual}")
    weights = torch.load(paths["weights"], map_location="cpu", weights_only=True)
    checkpoint = torch.load(paths["full"], map_location="cpu", weights_only=True)
    if checkpoint["artifact_sha256"] != WEIGHTS_SHA:
        raise ValueError("Full checkpoint does not identify these weights")
    original = checkpoint["model_artifact"]
    for field in ("schema", "model_kind", "model_config", "input_contract"):
        if original[field] != weights[field]:
            raise ValueError(f"Full checkpoint / weights metadata differ: {field}")
    if set(original["state_dict"]) != set(weights["state_dict"]):
        raise ValueError("Full checkpoint / weights state keys differ")
    for key, value in original["state_dict"].items():
        torch.testing.assert_close(value, weights["state_dict"][key], rtol=0, atol=0)
    manifest = json.loads(paths["manifest"].read_text())
    sizes = {(r["clip"]["height"], r["clip"]["width"]) for r in manifest["clips"]}
    if sizes != {(720, 1280)} or manifest["config"]["window_length"] != 32:
        raise ValueError("Pinned evaluation input geometry differs from the deployment")
    rows = [json.loads(line) for line in paths["frames"].read_text().splitlines()]
    calibration = calibrate(rows, split="val")
    validation = json.loads((args.snapshot / "validation-update-00010000.json").read_text())
    if any(validation["calibration"][key] != value for key, value in calibration.items()):
        raise ValueError("Recalculated threshold differs from training validation")
    metrics = frame_metrics(rows, calibration, distance_px=20.)
    weights["scene_input_contract"] = scene_dpt_input_contract(image_size=(720, 1280), frames=32, input_channels=5)
    weights["scene_export_source"] = dict(weights_sha256=WEIGHTS_SHA, full_checkpoint_sha256=FULL_SHA,
                                           evaluation_manifest_sha256=MANIFEST_SHA)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint.exists():
        raise FileExistsError(args.checkpoint)
    torch.save(weights, args.checkpoint)
    report = dict(schema="scene_x3d_dpt_export.v1", input_sha256=actual,
                  full_checkpoint_and_weights_state_identical=True,
                  state_tensors=len(weights["state_dict"]), geometry_from="pinned evaluation manifest, every clip",
                  scene_input_contract=weights["scene_input_contract"],
                  exported_checkpoint=str(args.checkpoint), exported_sha256=sha(args.checkpoint),
                  calibration=calibration, metrics=metrics,
                  evaluation_note="Calibrated and measured on the same validation; not an independent test.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(threshold=calibration["threshold"], fpr=metrics["negative_fpr"],
                          fnr=metrics["observed_fnr"], exported_sha256=report["exported_sha256"])))


if __name__ == "__main__":
    main()
