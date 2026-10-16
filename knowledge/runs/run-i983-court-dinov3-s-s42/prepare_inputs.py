import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

commit = "98b737829ba8baae8b823c7b7728cf0608f5073f"
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
    or item["destination"].endswith("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
]
if len(items) != 6:
    raise RuntimeError("Expected five datasets and the S pretrained weight")
output = Path("outputs/court_detection/prepare/i983-inputs/verified-03")
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
