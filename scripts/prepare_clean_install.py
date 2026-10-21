#!/usr/bin/env python3
"""Rehearse docs/installation in an independent clone; stop before GPU use.

Run with system python3. Every executed Python project command uses the newly
synced clone's .venv. User-acquired third-party files are read from an explicit
checkpoint root, verified, copied (never linked), and verified again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_local_assets(source_root: Path, destination_root: Path, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify the entire local selection before copying; never follow an output symlink."""
    pairs = []
    errors = []
    for asset in assets:
        if not asset["local_acquisition"]:
            continue
        filename = Path(asset["filename"])
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError(f"Catalog path must be checkpoint-root relative: {filename}")
        original, target = source_root / filename, destination_root / filename
        if target.exists() or target.is_symlink() or not target.resolve().is_relative_to(destination_root.resolve()):
            errors.append(f"Conflicting or escaping destination: {target}")
            continue
        if not original.is_file():
            errors.append(f"Missing user-acquired asset: {original}")
            continue
        actual = file_hash(original)
        if actual != asset["sha256"]:
            errors.append(f"SHA-256 mismatch: {original}; expected {asset['sha256']}, got {actual}")
            continue
        pairs.append((asset, original, target))
    if errors:
        raise ValueError("\n".join(errors))
    records = []
    for asset, original, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        actual = file_hash(target)
        if actual != asset["sha256"]:
            raise ValueError(f"Copied asset failed SHA-256 validation: {target}")
        records.append({"source": str(original), "destination": str(target), "sha256": actual,
                        "bytes": target.stat().st_size, "acquisition": "copy of locally user-acquired file",
                        "official_source": asset["source"]})
    return records


def gpu_commands(repo: Path, build: Path) -> tuple[str, str]:
    """Return shell-safe, explicit commands for the sample and its visualization."""
    prefix = f"cd {shlex.quote(str(repo))} && "
    python = shlex.quote(str(repo / ".venv/bin/python"))
    pipeline = (f"{prefix}PYTHONPATH={shlex.quote(str(build / 'lib'))} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
                f"{python} -m src.tennis_scene.scripts.run_pipeline "
                "output_directory=tennis_scene/generate/clean_install output_name=scene "
                "cache.directory=tennis_scene/cache/clean_install")
    visualization = (f"{prefix}CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg {python} "
                     "-m src.tennis_scene.scripts.visualization "
                     "input=tennis_scene/generate/clean_install/scene.npz "
                     "output=tennis_scene/visualize/clean_install/scene.mp4 "
                     "style.player_representation=smpl")
    return pipeline, visualization


def prepare(args: argparse.Namespace) -> None:
    workspace: Path = args.workspace.resolve()
    if workspace.exists():
        raise FileExistsError(f"Use a new workspace; existing runs are never overwritten: {workspace}")
    source_root: Path = args.local_checkpoint_root.resolve(strict=True)
    workspace.mkdir(parents=True)
    repo, build = workspace / "repo", workspace / "repo/outputs/dino_extension"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "CUDA_HOME": str(args.cuda_home.resolve(strict=True)), "TORCH_CUDA_ARCH_LIST": args.cuda_arch,
           "MAX_JOBS": "2", "UV_CACHE_DIR": str(workspace / "cache/uv"),
           "HF_HOME": str(workspace / "cache/huggingface"), "HF_HUB_DISABLE_PROGRESS_BARS": "1",
           "TORCH_HOME": str(workspace / "cache/torch"), "XDG_CACHE_HOME": str(workspace / "cache"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    # Do not inherit an extension or a Python environment from the caller.
    for variable in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME"):
        env.pop(variable, None)
    report: dict[str, Any] = {"status": "running", "gpu_executed": False, "steps": [], "workspace": str(workspace)}
    started = time.monotonic()

    def save() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        (workspace / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name: str, command: list[str], *, cwd: Path = workspace) -> str:
        step: dict[str, Any] = {"name": name, "command": command, "cwd": str(cwd)}
        report["steps"].append(step)
        save()
        print(f"[{name}] {shlex.join(command)}", flush=True)
        tick = time.monotonic()
        log = workspace / f"{name}.log"
        with log.open("w") as output:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=False)
        step.update(seconds=time.monotonic() - tick, exit_code=result.returncode, log=str(log))
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit {result.returncode}; see {log}")
        return log.read_text()

    try:
        run("clone", ["git", "clone", "--no-local", "--no-checkout", args.repository, str(repo)])
        run("checkout", ["git", "checkout", "--detach", args.revision], cwd=repo)
        report["commit"] = run("commit", ["git", "rev-parse", "HEAD"], cwd=repo).strip()
        # No inherited symlink to the main environment is ever installed.
        if (repo / ".venv").exists() or (repo / ".venv").is_symlink():
            raise FileExistsError("Fresh clone unexpectedly contains .venv")
        run("uv-sync", ["uv", "sync", "--locked"], cwd=repo)
        python = str(repo / ".venv/bin/python")
        run("python-environment", [python, "-c",
            "import sys,torch; from pathlib import Path; "
            "assert Path(sys.prefix) == Path.cwd()/'.venv'; assert not Path('.venv').is_symlink(); "
            "print(sys.prefix); print(torch.__version__,torch.version.cuda)"], cwd=repo)
        run("ffmpeg", ["ffmpeg", "-version"], cwd=repo)
        run("submodules", ["git", "submodule", "update", "--init", "third_party/DINO", "third_party/dinov3"], cwd=repo)
        run("build-dino", ["bash", "scripts/build_dino_extension.sh", str(repo), str(build)], cwd=repo)
        catalog = run("asset-catalog", [python, "-c",
            "import json; from dataclasses import asdict; from src.tennis_scene.installation_assets import INSTALLATION_ASSETS; "
            "print(json.dumps([asdict(a) for a in INSTALLATION_ASSETS]))"], cwd=repo)
        tick = time.monotonic()
        report["local_asset_copies"] = copy_local_assets(source_root, repo / "ckpt", json.loads(catalog))
        report["steps"].append({"name": "copy-user-acquired-assets", "seconds": time.monotonic() - tick, "exit_code": 0})
        save()
        run("download-models", [python, "-m", "src.utils.model_weights", "--checkpoint-root", str(repo / "ckpt")], cwd=repo)
        run("download-sample", [python, "-m", "src.utils.sample_dataset", "--data-root", str(repo / "data")], cwd=repo)
        env["PYTHONPATH"] = str(build / "lib")
        run("preflight", [python, "-m", "src.tennis_scene.scripts.preflight"], cwd=repo)
        run("git-status", ["git", "status", "--short", "--untracked-files=all"], cwd=repo)
        pipeline, visualization = gpu_commands(repo, build)
        for filename, command in (("run_pipeline.sh", pipeline), ("visualize.sh", visualization)):
            (workspace / filename).write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
        report.update(status="ready_for_gpu", pipeline_command=pipeline, visualization_command=visualization,
                      disk_apparent_bytes=int(subprocess.check_output(["du", "-sb", str(workspace)], text=True).split()[0]),
                      disk_allocated_bytes=int(subprocess.check_output(["du", "-sB1", str(workspace)], text=True).split()[0]))
        save()
        print(f"Ready for the parent to enqueue (GPU has not run):\nbash {shlex.quote(str(workspace / 'run_pipeline.sh'))}")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repository", default="https://github.com/Motoki0705/tennis-lab.git")
    parser.add_argument("--revision", required=True, help="Commit to rehearse; it must exist in the clone")
    parser.add_argument("--local-checkpoint-root", type=Path, required=True, help="User-acquired assets for this rehearsal only")
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--cuda-arch", required=True, help="Explicit TORCH_CUDA_ARCH_LIST, e.g. 12.0 for RTX 5060 Ti")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
