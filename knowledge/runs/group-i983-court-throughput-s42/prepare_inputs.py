import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

commit = "c4b12f923b243da6b502149aa6f3912c706045e7"
observed = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if observed != commit:
    raise RuntimeError(f"Unexpected source commit: {observed}")
manifest = Path(".i983_inputs.json")
if (
    hashlib.sha256(manifest.read_bytes()).hexdigest()
    != "4d75a2256088a12f8112cf1718d68bd3c649d71f64ad61429b66dba0add8dc7e"
):
    raise RuntimeError("Input manifest changed")
items = json.loads(manifest.read_text())
items = [
    item
    for item in items
    if "manifest_sha256" in item
    or item["destination"].endswith(("dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"))
]
if len(items) != 7:
    raise RuntimeError("Expected five datasets and the S+/B pretrained weights")
output = Path("outputs/court_detection/prepare/i983-inputs/verified-05")
output.mkdir(parents=True, exist_ok=False)
shutil.copyfile(".i983_inputs.json", output / "inputs.json")
for index, item in enumerate(items):
    destination = item["destination"]
    print(
        json.dumps(
            {"asset": index, "destination": destination, "status": "downloading"}
        ),
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            ".agents/skills/tennis-drive/scripts/drive.py",
            "download",
            destination,
            destination,
            "--verify",
        ],
        check=True,
    )


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


verified = []
for item in items:
    path = Path(item["destination"])
    if "manifest_sha256" in item:
        if digest(path / "dataset.json") != item["manifest_sha256"]:
            raise RuntimeError(f"Manifest mismatch: {path}")
        base = (
            path / "samples"
            if item["destination"].startswith("data/synthetic_data_generation/")
            else path
        )
        for rel, expected in item["files"]:
            if digest(base / rel) != expected:
                raise RuntimeError(f"Asset mismatch: {base / rel}")
    elif digest(path) != item["sha256"]:
        raise RuntimeError(f"Weight mismatch: {path}")
    verified.append(item["destination"])
receipt = {
    "status": "all_selected_inputs_verified",
    "commit": commit,
    "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "assets": verified,
}
(output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)

# Recover only the two VM-local checkpoints needed for S+ selection/evaluation.
# No checkpoint is copied to the operator's local workstation.
train_root = Path("outputs/court_detection/train/i983-dino-splus-shared1024/s42")
last_path = train_root / "logs/version_0/checkpoints/last.ckpt"
subprocess.run(
    [sys.executable, ".agents/skills/tennis-drive/scripts/drive.py", "download",
     str(last_path), str(last_path), "--verify"],
    check=True,
)
checkpoint = torch.load(last_path, map_location="cpu", mmap=True, weights_only=False)
if checkpoint["epoch"] != 19 or checkpoint["global_step"] != 33160:
    raise RuntimeError("S+ did not finish the approved 20 epochs")
callbacks = []
selected = []
for key, state in checkpoint["callbacks"].items():
    if not str(key).startswith("ModelCheckpoint"):
        continue
    score = state.get("best_model_score")
    row = {
        "key": key,
        "best_model_path": state.get("best_model_path"),
        "best_score": None if score is None else float(score),
        "ranking": {str(k): float(v) for k, v in state.get("best_k_models", {}).items()},
    }
    callbacks.append(row)
    if "'monitor': 'val/loss'" in str(key) and "'mode': 'min'" in str(key):
        selected.append(row)
if len(selected) != 1:
    raise RuntimeError("Expected exactly one val/loss min callback")
best = selected[0]
if best["best_score"] != min(best["ranking"].values()):
    raise RuntimeError("Saved best score disagrees with the candidate ranking")
best_relative = Path(best["best_model_path"]).relative_to(Path.cwd())
if not best_relative.is_relative_to(train_root):
    raise RuntimeError("Selected checkpoint escaped the S+ run")
subprocess.run(
    [sys.executable, ".agents/skills/tennis-drive/scripts/drive.py", "download",
     str(best_relative), str(best_relative), "--verify"],
    check=True,
)
selection = {
    "criterion": {"monitor": "val/loss", "mode": "min"},
    "final_epoch": checkpoint["epoch"],
    "final_global_step": checkpoint["global_step"],
    "best_checkpoint": best["best_model_path"],
    "best_score": best["best_score"],
    "ranking": best["ranking"],
    "checkpoint_sha256": digest(best_relative),
}
(output / "checkpoint-selection.json").write_text(json.dumps(selection, indent=2) + "\n")
(output / "checkpoint-callbacks.json").write_text(
    json.dumps({"epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
                "callbacks": callbacks}, indent=2) + "\n"
)
print(json.dumps(selection), flush=True)
