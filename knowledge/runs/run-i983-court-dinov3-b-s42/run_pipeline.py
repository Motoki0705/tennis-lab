"""Run B provisioning and the approved training consecutively in one Colab kernel job."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

MANIFEST = '[\n  {\n    "source": "/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B00/datasets/court",\n    "destination": "data/synthetic_data_generation/scenes/B00/datasets/court",\n    "manifest_sha256": "f2e37e8ad6df7a1b944ec34580b60c7c26f315ebb2b803273d1e3fd0a080e6ad",\n    "files": [\n      [\n        "index.npz",\n        "24721bcc3e621b536fd0673d606143384eca941d1052dcf6773e2a4d60d57e63"\n      ],\n      [\n        "shards/images-00000.bin",\n        "c8d14485fd68e2d8e50532ce7b7908f0b97f2aaf7abf8e91a15507b6871173f7"\n      ]\n    ]\n  },\n  {\n    "source": "/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B01/datasets/court",\n    "destination": "data/synthetic_data_generation/scenes/B01/datasets/court",\n    "manifest_sha256": "a2f7ee3a04bb3c6e39732b7917a1af5af0ad093446b36a0f7974e4569f436f4e",\n    "files": [\n      [\n        "index.npz",\n        "422b18d17e38b47560daf055e21c9562de96662ecef5dee0316feaa38b5e40cd"\n      ],\n      [\n        "shards/images-00000.bin",\n        "90ccd58fcadd3fe43bee040a7dae3adb8a7710440a1b10fe08d9cd71363ee451"\n      ]\n    ]\n  },\n  {\n    "source": "/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B02/datasets/court",\n    "destination": "data/synthetic_data_generation/scenes/B02/datasets/court",\n    "manifest_sha256": "40f17dd3345570c5f4897d2ee454909a678583e0a8237f4dca352e34cfc9b568",\n    "files": [\n      [\n        "index.npz",\n        "c6761029a4b20132e8c0eb873c6945e518a608f8f9bc0dff8c361afe901c062b"\n      ],\n      [\n        "shards/images-00000.bin",\n        "31f90f965e43be72712956db140755f544569c81338a26e54c77e0f90ee7fd14"\n      ]\n    ]\n  },\n  {\n    "source": "/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B03/datasets/court",\n    "destination": "data/synthetic_data_generation/scenes/B03/datasets/court",\n    "manifest_sha256": "84c8195ac34c4cf0a9eb4852ec85e37ce7940a58cc2b29a7637c1fa087d12607",\n    "files": [\n      [\n        "index.npz",\n        "062dceb390c2fbb1be86540bd2637623860ce3e2d5dc4a0306c105a19ae4807d"\n      ],\n      [\n        "shards/images-00000.bin",\n        "cc1f59f745bed650c2a4d40248d6b9e56d11ba4f90b3b86a0f68ef67219e52f3"\n      ]\n    ]\n  },\n  {\n    "source": "/home/kamimura/projects/tennis-lab/data/court_detection/tennis_court_detector-v1",\n    "destination": "data/court_detection/tennis_court_detector-v1",\n    "manifest_sha256": "fd0aa741a66ce6f47f85cadc798658be7a7a06a7729f21adf324f8b360771f41",\n    "files": [\n      [\n        "index.npz",\n        "31c1f71e57f4575bd0d72491fa65af31467e226c84045fcfaa0784bf7c13982b"\n      ],\n      [\n        "shards/images-00000.bin",\n        "20e190037abe7bf91cb06c556814b2d41348750894d52784208878abd2dabf05"\n      ],\n      [\n        "shards/images-00001.bin",\n        "ecfb50d97ee343a40d516921a5f53ef172a0a433e358fb348e93bf4b79f475f9"\n      ],\n      [\n        "shards/images-00002.bin",\n        "42aa2cc096bb4c0ba68d52e29e6b188ac478c06750ae7823755c8495240fe173"\n      ],\n      [\n        "shards/images-00003.bin",\n        "f6d362b234739f0587407c79692bbff9a559afd33b7bacda9d51845c6a9c5831"\n      ],\n      [\n        "shards/images-00004.bin",\n        "62bbb5e9016fe79e4e1a35815e0e6249c76ef1b3d95be5433d9c24dffef22d1d"\n      ],\n      [\n        "shards/images-00005.bin",\n        "5d8cc6725736fd6e9b3b0667e9c7224152962322a4a50e704e9116c353955d5d"\n      ],\n      [\n        "shards/images-00006.bin",\n        "53e33c7a2cb3160b641c5a40dda5c9cd3af84d49be40cf5b2d31b64f1be4b2f3"\n      ],\n      [\n        "shards/images-00007.bin",\n        "62ee6757833d30b48320b44c00fe335f809cfafceb44b2fae5a3267b176c92ec"\n      ]\n    ]\n  },\n  {\n    "source": "/home/kamimura/projects/tennis-lab/ckpt/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",\n    "destination": "ckpt/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",\n    "sha256": "08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d",\n    "bytes": 86531063\n  },\n  {\n    "source": "/home/kamimura/projects/tennis-lab/ckpt/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",\n    "destination": "ckpt/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",\n    "sha256": "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea",\n    "bytes": 114924783\n  },\n  {\n    "source": "/home/kamimura/projects/tennis-lab/ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",\n    "destination": "ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",\n    "sha256": "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",\n    "bytes": 342860279\n  },\n  {\n    "source": "/home/kamimura/projects/tennis-lab/ckpt/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",\n    "destination": "ckpt/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",\n    "sha256": "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035",\n    "bytes": 1213050671\n  }\n]'
PREPARE = 'import hashlib\nimport json\nimport shutil\nimport subprocess\nimport sys\nfrom pathlib import Path\n\ncommit = "75e30a565b1f7a5f7fd4f08cea2daf458d377653"\nobserved = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()\nif observed != commit:\n    raise RuntimeError(f"Unexpected source commit: {observed}")\nmanifest = Path(".i983_inputs.json")\nif (\n    hashlib.sha256(manifest.read_bytes()).hexdigest()\n    != "4d75a2256088a12f8112cf1718d68bd3c649d71f64ad61429b66dba0add8dc7e"\n):\n    raise RuntimeError("Input manifest changed")\nitems = json.loads(manifest.read_text())\nitems = [\n    item\n    for item in items\n    if "manifest_sha256" in item\n    or item["destination"].endswith("dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")\n]\nif len(items) != 6:\n    raise RuntimeError("Expected five datasets and the B pretrained weight")\noutput = Path("outputs/court_detection/prepare/i983-inputs/verified-07")\noutput.mkdir(parents=True, exist_ok=False)\nshutil.copyfile(".i983_inputs.json", output / "inputs.json")\nfor index, item in enumerate(items):\n    destination = item["destination"]\n    print(\n        json.dumps(\n            {"asset": index, "destination": destination, "status": "downloading"}\n        ),\n        flush=True,\n    )\n    subprocess.run(\n        [\n            sys.executable,\n            ".agents/skills/tennis-drive/scripts/drive.py",\n            "download",\n            destination,\n            destination,\n            "--verify",\n        ],\n        check=True,\n    )\n\n\ndef digest(path, algorithm="sha256"):\n    with path.open("rb") as stream:\n        return hashlib.file_digest(stream, algorithm).hexdigest()\n\n\nverified = []\nfor item in items:\n    path = Path(item["destination"])\n    if "manifest_sha256" in item:\n        if digest(path / "dataset.json") != item["manifest_sha256"]:\n            raise RuntimeError(f"Manifest mismatch: {path}")\n        base = (\n            path / "samples"\n            if item["destination"].startswith("data/synthetic_data_generation/")\n            else path\n        )\n        for rel, expected in item["files"]:\n            if digest(base / rel) != expected:\n                raise RuntimeError(f"Asset mismatch: {base / rel}")\n    elif digest(path) != item["sha256"]:\n        raise RuntimeError(f"Weight mismatch: {path}")\n    verified.append(item["destination"])\nreceipt = {\n    "status": "all_selected_inputs_verified",\n    "commit": commit,\n    "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),\n    "assets": verified,\n}\n(output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\\n")\nprint(json.dumps(receipt), flush=True)\n'
COMMIT = "75e30a565b1f7a5f7fd4f08cea2daf458d377653"
OUTPUT = Path("outputs/court_detection/prepare/i983-b-pipeline/session-07")


def record(name, data):
    temporary = OUTPUT.parent / ("." + OUTPUT.name + "-" + name + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(OUTPUT / name)


def now():
    return datetime.now(UTC).isoformat()


def main():
    if Path.cwd() != Path("/content/tennis-lab") or Path("/content/.tennis-colab") not in Path(os.environ.get("RCLONE_CONFIG", "")).parents:
        raise RuntimeError("Run only through the managed Colab session")
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if observed != COMMIT:
        raise RuntimeError("Scientific source differs from the approved commit")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    record("pipeline-start.json", {"started_at": now(), "commit": observed, "config_name": "train_i983_b"})
    subprocess.run(["uv", "sync", "--locked", "--python", "3.11"], check=True)
    subprocess.run(["git", "submodule", "update", "--init", "third_party/dinov3"], check=True)
    if hashlib.sha256(MANIFEST.encode()).hexdigest() != "4d75a2256088a12f8112cf1718d68bd3c649d71f64ad61429b66dba0add8dc7e":
        raise RuntimeError("Input manifest changed")
    Path(".i983_inputs.json").write_text(MANIFEST)
    env_code = "import importlib.metadata as m,json,os,platform,subprocess; print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'cpu_count':os.cpu_count(),'packages':{p:m.version(p) for p in ('torch','torchvision','pytorch-lightning','numpy','hydra-core')},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),'dinov3':subprocess.check_output(['git','submodule','status','third_party/dinov3'],text=True).strip(),'matrix_instrumentation_enabled':bool(os.environ.get('TENNIS_COURT_MATRIX_MANIFEST_PATH'))}))"
    environment = json.loads(subprocess.check_output([".venv/bin/python", "-c", env_code], text=True))
    if "NVIDIA L4" not in environment["gpu"]:
        raise RuntimeError("Expected L4")
    record("environment.json", {"recorded_at": now(), "git_commit": observed, **environment})
    subprocess.run([".venv/bin/python", "-c", PREPARE], check=True)
    argv = [".venv/bin/python", "-m", "src.tasks.court_detection.scripts.train", "--config-name", "train_i983_b", "run.artifact_store.mode=rclone", "run.artifact_store.remote=gdrive", "run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/i983-dino-b-shared1024/s42", "run.artifact_store.sync_interval_seconds=60"]
    process = subprocess.Popen(argv)
    receipt = {"training_command_started_at": now(), "pid": process.pid, "commit": observed, "argv": argv, "input_verification": "outputs/court_detection/prepare/i983-inputs/verified-07/verification.json"}
    record("training-start.json", receipt)
    print(json.dumps(receipt), flush=True)
    returncode = process.wait()
    record("training-end.json", {"training_command_finished_at": now(), "returncode": returncode})
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
