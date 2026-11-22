"""Acquire pinned capacity inputs and the C0 full checkpoint without remote writes."""
import argparse
import json
import subprocess
from pathlib import Path

from rebuild_dataset import sha, write

CONFIG = "/home/kamimura/.config/tennis-lab/rclone-drive.conf"
CLI = Path(".agents/skills/tennis-drive/scripts/drive.py")


def download_snapshot(root, relative, destination, marker_sha, label):
    metadata = root / "input-receipts"
    metadata.mkdir(exist_ok=True)
    marker = metadata / (label + "-SNAPSHOT.json")
    prefix = [str(Path('.venv/bin/python').absolute()), str(CLI), "--rclone-config", CONFIG]
    if not marker.exists():
        result = subprocess.run([*prefix, "download", relative + "/_SNAPSHOT.json", str(marker), "--verify"], capture_output=True, text=True)
        write(metadata / (label + "-marker-download.json"), dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        result.check_returncode()
    if sha(marker) != marker_sha:
        raise ValueError("Snapshot marker differs from saved receipt: " + label)
    raw = json.loads(marker.read_text())
    expected = {name: item["sha256"] for name, item in raw["files"].items()}
    expected["_SNAPSHOT.json"] = marker_sha
    path = metadata / (label + "-expected.json")
    write(path, dict(schema_version=1, files=expected))
    result = subprocess.run([*prefix, "--timeout-seconds", "7200", "download-tree", relative, str(destination), "--expected-manifest", str(path), "--progress-file", str(metadata / (label + "-progress.json"))], capture_output=True, text=True)
    write(metadata / (label + "-download.json"), dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    result.check_returncode()


def main(root):
    selected = json.loads((root / "restore-selected-expected.json").read_text())["files"]
    for name, digest in selected.items():
        if sha(root / "original" / name) != digest:
            raise ValueError("Selected architecture input differs: " + name)
    status = json.loads(Path('/mnt/e/tennis-lab-archive/metadata/ball-arch-20261113/source-uncovered/stage-status/T1-q06000-q12000/state.json').read_text())
    if status["update"] != 5500 or status["readback"]["status"] != "verified":
        raise ValueError("Expected the latest verified 5500 snapshot")
    checkpoint_dir = root / "original/T1-q06000-q12000/checkpoint-00005500-cddcef20"
    if sha(checkpoint_dir / "_SNAPSHOT.json") != status["receipt"]["manifest_sha256"]:
        raise ValueError("Checkpoint manifest differs from verified receipt")
    # The large historical source.bundle is already in Git and is not needed for
    # execution. Verify every selected metadata file against its pinned marker.
    marker = root / "capacity-marker.json"
    receipt = json.loads((root / "capacity-receipt.json").read_text())
    if sha(marker) != receipt["manifest_sha256"]:
        raise ValueError("Capacity input marker differs")
    for name, item in json.loads(marker.read_text())["files"].items():
        if name != "source.bundle" and sha(root / "inputs/capacity12k" / name) != item["sha256"]:
            raise ValueError("Capacity metadata differs: " + name)
    parent = json.loads((root / "inputs/capacity12k/parent-receipts/common-dpt-T32.json").read_text())
    relative = "outputs/ball_detection/train/capacity_scaling/s42-20261103-main/common-dpt-T32/" + parent["receipt"]["generation"]
    download_snapshot(root, relative, root / "inputs/common-dpt-T32", parent["receipt"]["manifest_sha256"], "c0")
    if sha(root / "inputs/common-dpt-T32/checkpoint-update-00010000.pt") != parent["sha256"]:
        raise ValueError("C0 checkpoint differs")
    write(root / "inputs-verified.json", dict(complete=True, original_checkpoint_update=5500,
          capacity_inputs_manifest_sha256=receipt["manifest_sha256"],
          c0_checkpoint_sha256=parent["sha256"], original_snapshot_status=status,
          source_bundle_reused_from_git="fae715ad752b2da41607d139ce6387d8d3f79823"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    main(parser.parse_args().root.resolve())
