"""Lightweight execution-image checks, run inside the sandbox (GPU via queue)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="automation.chatgpt_mcp.capabilities",
    fields=(
        BoundaryPathField(
            "lock",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
    ),
)


def check_vision() -> dict[str, Any]:
    import cv2
    import numpy as np
    from numpy.typing import NDArray

    binary = next(Path(cv2.__file__).parent.glob("cv2*.so"))
    dependencies = subprocess.run(
        ["ldd", str(binary)], capture_output=True, text=True, check=True
    ).stdout
    if "not found" in dependencies:
        raise RuntimeError(dependencies)
    pixels: NDArray[np.uint8] = np.full((32, 48, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", pixels)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
    if decoded is None or not np.array_equal(decoded, pixels):
        raise RuntimeError("PNG roundtrip failed")
    return {"opencv": cv2.__version__, "ldd_missing": False, "png_roundtrip": True}


def check_video() -> dict[str, Any]:
    import cv2

    with tempfile.TemporaryDirectory() as directory:
        movie = str(Path(directory) / "probe.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x48:r=5",
                "-t",
                "1",
                "-c:v",
                "mpeg4",
                movie,
            ],
            check=True,
        )
        info = json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", movie]
            )
        )
        capture = cv2.VideoCapture(movie)
        try:
            count = 0
            while capture.read()[0]:
                count += 1
        finally:
            capture.release()
        if count != 5:
            raise RuntimeError(f"expected 5 decoded frames, got {count}")
        return {"codec": info["streams"][0]["codec_name"], "frames": count}


def check_torch(gpu: bool) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    if not gpu and torch.cuda.is_available():
        raise RuntimeError("CPU sandbox unexpectedly exposes CUDA")
    devices = (
        [f"cuda:{i}" for i in range(torch.cuda.device_count())] if gpu else ["cpu"]
    )
    if not devices:
        raise RuntimeError("GPU sandbox exposes no CUDA device")
    results = []
    for device in devices:
        torch.manual_seed(42)
        model = torch.nn.Linear(4, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.ones(16, 4, device=device)
        target = torch.full((16, 1), 3.0, device=device)
        losses = []
        initial = model.weight.detach().clone()
        for _ in range(25):
            optimizer.zero_grad()
            loss = (model(x) - target).square().mean()
            loss.backward()
            if not torch.isfinite(loss) or any(
                p.grad is None or not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise RuntimeError("nonfinite loss or gradients")
            optimizer.step()
            losses.append(float(loss.detach()))
        if losses[-1] >= losses[0] or torch.equal(initial, model.weight):
            raise RuntimeError("optimizer failed to learn")
        results.append(
            {
                "device": device,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "steps": 25,
            }
        )
    return {"torch": torch.__version__, "runs": results}


def check_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(["git", "init", "-q", directory], check=True)
    return {"git": shutil.which("git"), "pytest": importlib.metadata.version("pytest")}


def check_compiler() -> dict[str, Any]:
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("C compiler executable 'cc' was not found")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "probe.c"
        executable = root / "probe"
        source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
        subprocess.run([compiler, str(source), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)
    version = subprocess.run(
        [compiler, "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()[0]
    return {"compiler": compiler, "version": version, "compile_and_run": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root
    roots = RuntimePathRoots(
        project_root=root,
        data_root=root,
        checkpoint_root=root,
        artifact_root=root,
        output_root=root,
        cache_root=root,
        external_asset_root=root,
    )
    PATH_BOUNDARY.validate({"lock": root / "uv.lock"}, resolver=PathResolver(roots))
    report: dict[str, Any] = {
        "profiles": {},
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
    }
    for name, check in (
        ("core_torch", lambda: check_torch(args.gpu)),
        ("vision", check_vision),
        ("video", check_video),
        ("compiler", check_compiler),
        ("test", check_test),
    ):
        try:
            report["profiles"][name] = {"ok": True, "details": check()}
        except Exception as error:
            report["profiles"][name] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    print(json.dumps(report, indent=2))
    return 0 if all(item["ok"] for item in report["profiles"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
