"""Reproduce the repository CUDA build with an explicit private interpreter."""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from rebuild_dataset import sha, write


def build(root, assets):
    original = json.loads((root / "original-preparation/msda-build-manifest.json").read_text())
    source = assets / "third_party/DINO/models/dino/ops/src"
    for name, digest in original["source_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError("DINO source differs from the official-verified original: " + name)
    script = Path("scripts/build_dino_extension.sh").read_text()
    assignment = 'python="${code_root}/.venv/bin/python"'
    if script.count(assignment) != 1:
        raise ValueError("Repository build interpreter selection changed")
    generated = script.replace(assignment, "python=" + shlex.quote(sys.executable))
    target = root / "msda-build"
    target.mkdir(exist_ok=True)
    wrapper = target / "build-with-private-python.sh"
    wrapper.write_text(generated)
    env = dict(os.environ, CUDA_HOME="/usr/local/cuda", TORCH_CUDA_ARCH_LIST="12.0",
               MAX_JOBS="1", PYTHONDONTWRITEBYTECODE="1", NVCC_APPEND_FLAGS="--maxrregcount=64")
    command = ["bash", str(wrapper), str(assets), str(target)]
    write(target / "invocation.json", dict(command=command, interpreter=sys.executable,
          original_script_sha256=hashlib.sha256(script.encode()).hexdigest(),
          generated_script_sha256=sha(wrapper), only_script_change="explicit private Python interpreter",
          NVCC_APPEND_FLAGS=env["NVCC_APPEND_FLAGS"],
          register_cap_reason="SM120 double backward used 66 registers/thread for 1024 threads; exceeds 65536 registers/block"))
    subprocess.run(command, env=env, check=True)
    import torch
    manifest = dict(original, schema="desktop_official_msda_build.v1",
                    original_build_manifest_sha256=sha(root / "original-preparation/msda-build-manifest.json"),
                    build=json.loads((target / "build.json").read_text()),
                    python=sys.version, device=torch.cuda.get_device_name(),
                    compute_capability=torch.cuda.get_device_capability(),
                    driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
                    compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                    binary_sha256=sha(target / "lib/MultiScaleDeformableAttention.so"),
                    official_source_sha256_exact=True, cuda_algorithm_changes=False,
                    extra_nvcc_flags=env["NVCC_APPEND_FLAGS"],
                    unbounded_build_resources="msda-build-sm120-unbounded/resources.txt",
                    command=command)
    write(root / "msda-build-manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(), args.assets.resolve())
