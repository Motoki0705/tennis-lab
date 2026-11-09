"""Ignored, bounded execution wrapper for the issue 1776 preview only."""

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

BASE = Path("outputs/tennis_scene/visualize/player_view/stabilization-comparison")
OPS = BASE / "operations/attempt2"
OUTPUT = BASE / "attempt2"
REF = "d133993421c206af28340f4ffe082c65b70fe7a7"
SESSION = "player-view-stabilization-20261109-a2"
BLENDER = Path("/content/blender/blender-5.2.1-linux-x64/blender")
URL = "https://download.blender.org/release/Blender5.2/blender-5.2.1-linux-x64.tar.xz"
SHA256 = "a31f524fa99a527d3d52b7f5aaa68c34e1a19d5a1c9473f79c5cc610fd5b10e9"
KEY = "issue1776/player-view-stabilization-20261109-a2/preview/attempt-2"


def now():
    return dt.datetime.now(dt.timezone.utc)


def stamp(value=None):
    return (value or now()).isoformat().replace("+00:00", "Z")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def budget():
    operation = read(OPS / "gpu-operation.json")
    deadline = dt.datetime.fromisoformat(operation["command_deadline_utc"].replace("Z", "+00:00"))
    seconds = math.floor((deadline - now()).total_seconds())
    if seconds < 60:
        raise RuntimeError("GPU command deadline reached; persistence/release reserve protected")
    return seconds


def run(argv, cap=None, output=None):
    limit = budget()
    bounded = ["timeout", "--signal=TERM", "--kill-after=20s", f"{max(1, limit - 20)}s", *map(str, argv)]
    print("COMMAND " + shlex.join(bounded), flush=True)
    if output:
        with Path(output).open("w") as handle:
            subprocess.run(bounded, check=True, stdout=handle)
    else:
        subprocess.run(bounded, check=True)


def source():
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if ref != REF:
        raise RuntimeError(f"Source ref mismatch: {ref}")
    return ref


def launch():
    local_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    compared_paths = ["src", "tests", "pyproject.toml", "uv.lock", "knowledge/runs/run-player-camera-stabilization-20261109/render_comparison.py"]
    subprocess.run(["git", "diff", "--exit-code", REF, "HEAD", "--", *compared_paths], check=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *compared_paths], check=True)
    receipt_path = Path("outputs/tennis_scene/visualize/player_view/stabilization-research/drive-input-upload.json")
    receipt = read(receipt_path)
    assert receipt.get("status") == "completed" and receipt.get("verified") is True and receipt.get("dry_run") is False, receipt
    write(OPS / "input-upload-verified.json", receipt)
    sys.path.insert(0, str(Path(".agents/skills/tennis-colab/scripts").resolve()))
    import colab

    original_save = colab.Session.save
    operation = {
        "schema_version": 1,
        "issue": 1776,
        "session": SESSION,
        "source_ref": REF,
        "local_launch_ref": local_ref,
        "attempt": 2,
        "model": "gpt-6.1-sol",
        "effort": "max",
        "provider": "codex",
        "agent_session": os.environ["CODEX_THREAD_ID"],
        "gpu_requested": "L4",
        "maximum_allocation_seconds": 14400,
        "persistence_release_reserve_seconds": 960,
        "idempotency_key": KEY,
        "input_upload_receipt": str(OPS / "input-upload-verified.json"),
        "input_upload_receipt_sha256": digest(receipt_path),
        "input_manifest_sha256": digest(BASE / "inputs/manifest.json"),
        "input_upload_verified": True,
        "launch_argv": [".venv/bin/python", ".agents/skills/tennis-colab/scripts/colab.py", "start", "--session", SESSION, "--ref", REF, "--gpu", "L4"],
        "phase": "launching",
    }

    def save_and_record(instance, **fields):
        state = original_save(instance, **fields)
        phase = fields.get("phase")
        if phase == "allocating":
            operation["allocation_request_utc"] = state["created_at"]
            requested = dt.datetime.fromisoformat(state["created_at"].replace("Z", "+00:00"))
            operation["deadline_utc"] = stamp(requested + dt.timedelta(seconds=14400))
            operation["command_deadline_utc"] = stamp(requested + dt.timedelta(seconds=14400 - 960))
        elif phase == "allocated":
            operation["allocation_utc"] = state["updated_at"]
            allocated = dt.datetime.fromisoformat(state["updated_at"].replace("Z", "+00:00"))
            operation["allocation_confirmed_utc"] = stamp(allocated)
            operation["allocation_time_basis"] = "Official new returned successfully; hard deadline conservatively uses request + 4h"
        elif phase == "ready":
            operation["bootstrap_observed"] = state["observed"]
        if phase:
            operation["phase"] = phase
            operation["updated_at"] = stamp()
            write(OPS / "gpu-operation.json", operation)
        return state

    colab.Session.save = save_and_record
    sys.argv = ["colab.py", "start", "--session", SESSION, "--ref", REF, "--gpu", "L4"]
    raise SystemExit(colab.main())


def setup():
    source()
    assert not subprocess.check_output(["git", "diff", "HEAD", "--name-only"], text=True).strip()
    assert read(OPS / "input-upload-verified.json")["verified"] is True
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
    assert "L4" in gpu, gpu
    run(["apt-get", "update", "-qq"], 600)
    run(["apt-get", "install", "-y", "ffmpeg", "xz-utils", "libxrender1", "libxi6", "libxkbcommon0", "libsm6", "libgl1", "libxext6", "libegl1", "libxxf86vm1", "libxfixes3"], 900)
    run(["uv", "sync", "--locked", "--no-group", "gpu", "--group", "cpu"], 3600)
    archive = Path("/content/blender/blender-5.2.1-linux-x64.tar.xz")
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        run(["curl", "--fail", "--location", "--retry", "3", "--output", archive, URL], 1800)
    actual = digest(archive)
    if actual != SHA256:
        raise RuntimeError(f"Official Blender archive checksum mismatch: {actual}")
    if not BLENDER.exists():
        run(["tar", "-xJf", archive, "-C", archive.parent], 600)
    versions = {
        "recorded_at": stamp(),
        "source_ref": source(),
        "gpu": gpu,
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        "blender": subprocess.check_output([str(BLENDER), "--version"], text=True),
        "blender_url": URL,
        "blender_archive_sha256": actual,
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True),
        "python": subprocess.check_output([".venv/bin/python", "--version"], text=True).strip(),
        "python_executable": str(Path(".venv/bin/python").resolve()),
        "torch": json.loads(subprocess.check_output([".venv/bin/python", "-c", "import json, torch; print(json.dumps({'version':torch.__version__, 'cuda':torch.version.cuda}))"], text=True)),
        "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "renderer_backend": "OPTIX",
    }
    assert versions["blender"].startswith("Blender 5.2.1"), versions["blender"]
    assert versions["torch"]["cuda"] is None, versions["torch"]
    assert "--enable-libx264" in versions["ffmpeg"]
    write(OPS / "versions.json", versions)
    target = BASE / "inputs"
    if target.exists():
        raise FileExistsError("Input target unexpectedly exists; inspect instead of duplicating download")
    run([".venv/bin/python", ".agents/skills/tennis-drive/scripts/drive.py", "--timeout-seconds", str(min(3600, budget() - 30)), "download", str(target), str(target), "--verify"], output=OPS / "drive-input-download.json")
    receipt = read(OPS / "drive-input-download.json")
    assert receipt.get("status") == "completed" and receipt.get("verified") is True and receipt.get("dry_run") is False, receipt
    assert digest(target / "manifest.json") == read(OPS / "gpu-operation.json")["input_manifest_sha256"]
    manifest = read(target / "manifest.json")
    assert manifest["git_ref"] == REF
    assert manifest["preview_source_frames"] == [0, 26, 162]
    assert manifest["player_ids"] == [0, 1]
    assert [p["sigma_seconds"] for p in manifest["packages"]] == [0.0, 0.06, 0.08, 0.12]
    for item in manifest["packages"]:
        package = target / "packages" / item["name"]
        assert digest(package / "manifest.json") == item["manifest_sha256"]
        for filename, expected in read(package / "manifest.json")["files"].items():
            assert digest(package / filename) == expected, filename
    write(OPS / "setup-complete.json", {"completed_at": stamp(), "source_ref": source(), "gpu": gpu, "verified_input_receipt": str(OPS / "drive-input-download.json"), "verified": True})
    print("SETUP_COMPLETE inputs_verified=true", flush=True)


def audit():
    sys.path.insert(0, str(Path.cwd()))
    import cv2
    from src.tennis_scene.visualization.player_view.video import validate_render

    inputs = read(BASE / "inputs/manifest.json")
    preview = OUTPUT / "preview"
    manifest = read(preview / "manifest.json")
    assert manifest["phase"] == "preview" and manifest["source_frames"] == [0, 26, 162]
    assert manifest["git_ref"] == REF
    assert len(manifest["variants"]) == 4
    rows = []
    selected = [preview / "manifest.json", preview / "checkpoint.json"]
    counts = {0: 0, 1: 0}
    for item in inputs["packages"]:
        name = item["name"]
        package = BASE / "inputs/packages" / name
        videos_manifest = preview / "videos" / name / "manifest.json"
        published = read(videos_manifest)
        assert published["preview"] is True and published["source_frames"] == [0, 26, 162]
        assert published["display_lens"] == read(package / "manifest.json")["lens"]
        assert published["human_acceptance"] == "pending"
        selected.append(videos_manifest)
        for player in [0, 1]:
            render = preview / "renders" / name / f"player_{player}"
            report = validate_render(package, render, player)
            assert [r["source_frame"] for r in report["frames"]] == [0, 26, 162]
            assert report["gpu_devices"] and report["engine"] == "CYCLES"
            assert all(device["type"] == "OPTIX" and "L4" in device["name"] for device in report["gpu_devices"])
            assert report["camera"]["depth_of_field"] is False
            assert report["camera"]["resolution"] == [2560, 1440]
            assert len(list((render / "frames").glob("*.png"))) == 3
            counts[player] += 3
            rows.append({"variant": name, "player_id": player, "gpu_devices": report["gpu_devices"], "source_frames": [r["source_frame"] for r in report["frames"]], "missing_camera_frames": [r["source_frame"] for r in report["frames"] if not r["camera_valid"]], "render_seconds": sum(r["render_seconds"] for r in report["frames"])})
            selected.append(render / "render.json")
    assert counts == {0: 12, 1: 12}
    assert len(list((preview / "renders").glob("**/frames/*.png"))) == 24
    for player in [0, 1]:
        path = preview / f"player_{player}_contact.png"
        image = cv2.imread(str(path))
        assert image is not None and image.shape == (1080, 2560, 3), (path, None if image is None else image.shape)
        selected.append(path)
    validation = {"schema_version": 1, "validated_at": stamp(), "source_ref": source(), "phase": "preview", "source_frames": [0, 26, 162], "images_per_person": counts, "total_source_pngs": 24, "elapsed_seconds": manifest["elapsed_seconds"], "renders": rows, "human_acceptance": "pending"}
    write(preview / "validation.json", validation)
    selected.append(preview / "validation.json")
    recovery = {"schema_version": 1, "created_at": stamp(), "files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p), "md5": digest(p, "md5")} for p in selected]}
    write(preview / "recovery-manifest.json", recovery)
    print("PREVIEW_VALIDATED " + json.dumps(validation), flush=True)


def preview():
    source()
    assert read(OPS / "setup-complete.json")["verified"] is True
    if (OPS / "queue-registration.json").exists():
        raise RuntimeError("Preview already registered; inspect the exact existing job")
    if (OUTPUT / "preview").exists():
        raise FileExistsError(OUTPUT / "preview")
    operation = read(OPS / "gpu-operation.json")
    seconds = budget() - 90
    assert seconds >= 60
    renderer = [".venv/bin/python", "knowledge/runs/run-player-camera-stabilization-20261109/render_comparison.py", "--phase", "preview", "--inputs", str(BASE / "inputs"), "--output", str(OUTPUT / "preview"), "--blender", str(BLENDER)]
    command = shlex.join(["timeout", "--signal=TERM", "--kill-after=20s", f"{seconds}s", *renderer])
    queue = Path(".agents/skills/training-queue/scripts/training_queue.sh")
    queue_dir = (OPS / "training-queue").resolve()
    os.environ["TRAINING_QUEUE_DIR"] = str(queue_dir)
    add = ["bash", str(queue), "add", command, "--name", "issue1776_preview", "--provider", "codex", "--session", operation["agent_session"], "--issue", "1776", "--resource", "all", "--idempotency-key", KEY]
    write(OPS / "queue-command.json", {"created_at": stamp(), "command": command, "renderer_argv": renderer, "add_argv": add, "timeout_seconds": seconds, "source_ref": REF, "training_queue_dir": str(queue_dir)})
    receipt = subprocess.check_output(add, text=True)
    (OPS / "queue-add.txt").write_text(receipt)
    match = re.search(r"^queued: (\S+\.job)$", receipt, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("Queue registration did not return exactly one job identity")
    job = match.group(1)
    write(OPS / "queue-registration.json", {"registered_at": stamp(), "job": job, "idempotency_key": KEY, "provider": "codex", "session": operation["agent_session"], "resource": "all", "issue": 1776, "source_ref": REF, "command": command})
    print("PREVIEW_QUEUED " + job, flush=True)
    serve = subprocess.run(["bash", str(queue), "serve", "--idle-timeout", "1"])
    states = [state for state in ["jobs", "running", "done", "failed", "cancelled"] if (queue_dir / state / job).exists()]
    log = queue_dir / "logs" / (job[:-4] + ".log")
    repro = read(queue_dir / "repro" / job[:-4] / "run.json")
    result = {"completed_at": stamp(), "job": job, "states": states, "serve_returncode": serve.returncode, "log": str(log), "repro": str(queue_dir / "repro" / job[:-4]), "source_ref": REF}
    write(OPS / "queue-result.json", result)
    assert serve.returncode == 0 and states == ["done"], result
    assert re.search(r"^exit_code=0$", log.read_text(), flags=re.MULTILINE)
    assert repro["commit"] == REF, repro
    audit()
    print("PREVIEW_QUEUE_COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    OPS.mkdir(parents=True, exist_ok=True)
    {"launch": launch, "setup": setup, "preview": preview, "audit": audit}[sys.argv[1]]()
