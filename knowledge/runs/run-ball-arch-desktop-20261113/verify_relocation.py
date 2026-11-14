"""Verify relocated hardlinks, recorded private packages, native binary and cache."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from guard import D_DRIVE, E_DRIVE, d_drive_available, e_drive_available
from rebuild_dataset import DISTRIBUTION, sha, write


def verify(root, previous, *, hot_dataset=None):
    import av
    import cv2
    import numpy
    import torch

    started = time.monotonic()
    dataset = json.loads((root / "dataset-verified.json").read_text())
    if not (dataset["complete"] and dataset["all_sha256_verified"]
            and dataset["distribution_sha256"] == DISTRIBUTION
            and dataset["verified_files"] == 2925
            and dataset["verified_bytes"] == 157028755547):
        raise ValueError("Full relocated dataset verification is incomplete")
    volume_device = E_DRIVE.stat().st_dev
    if not root.resolve().is_relative_to(E_DRIVE) or root.stat().st_dev != volume_device:
        raise ValueError("New output root must be on the mounted data volume")
    entries = json.loads((root / "dataset/distribution.json").read_text())["files"]
    canonical = previous / "dataset" if hot_dataset is None else Path(dataset["source"])
    if hot_dataset is not None:
        if ((root / "dataset").resolve() != hot_dataset.resolve()
                or hot_dataset.stat().st_dev != Path("/").stat().st_dev
                or hot_dataset.resolve().is_relative_to(E_DRIVE)
                or not canonical.resolve().is_relative_to(E_DRIVE)
                or not dataset.get("independent_copy")):
            raise ValueError("Hot dataset must be a verified root-filesystem copy of the canonical E dataset")
    for entry in entries:
        source, target = canonical / entry["path"], root / "dataset" / entry["path"]
        if hot_dataset is not None:
            if os.path.samefile(source, target) or target.stat().st_nlink != 1 or target.stat().st_dev != Path("/").stat().st_dev:
                raise ValueError("Hot payload is not an independent root-filesystem copy")
        elif not os.path.samefile(source, target) or target.stat().st_dev != volume_device:
            raise ValueError("Dataset hardlink/device differs: " + str(target))
    inputs = json.loads((root / "inputs-verified.json").read_text())
    if sha(root / "inputs/common-dpt-T32/checkpoint-update-00010000.pt") != inputs["c0_checkpoint_sha256"]:
        raise ValueError("C0 checkpoint differs from the original verified input receipt")
    expected = json.loads((root / "environment-verified.json").read_text())
    differences = {}
    for package, version in expected["installed"].items():
        actual = importlib.metadata.version(package)
        if actual != version:
            differences[package] = dict(expected=version, actual=actual)
    if differences or sys.version != expected["python"] or torch.__version__ != expected["torch"]:
        raise ValueError("Relocated private environment differs: " + str(differences))
    build = json.loads((root / "msda-build-manifest.json").read_text())
    binary = root / "msda-build/lib/MultiScaleDeformableAttention.so"
    digest = sha(binary)
    if digest != build["binary_sha256"]:
        raise ValueError("Native MSDA binary differs")
    spec = importlib.util.spec_from_file_location("MultiScaleDeformableAttention", binary)
    if spec is None or spec.loader is None:
        raise ValueError("Native MSDA loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not all(hasattr(module, key) for key in ("ms_deform_attn_forward", "ms_deform_attn_backward")):
        raise ValueError("Native MSDA entrypoints missing")
    caches = {}
    for name in ("inductor", "triton"):
        target, old = root / "cache" / name, previous / "cache" / name
        if not target.is_dir() or not os.path.samefile(target, old) or not any(target.iterdir()):
            raise ValueError("Relocated compilation cache unavailable: " + name)
        caches[name] = str(target.resolve())
    assets = []
    for name in ("original", "original-preparation", "inputs"):
        for source in (previous / name).rglob("*"):
            if source.is_file():
                target = root / source.relative_to(previous)
                if not os.path.samefile(source, target):
                    raise ValueError("Relocated input hardlink differs: " + str(target))
                assets.append(dict(path=str(target.relative_to(root)), size_bytes=target.stat().st_size,
                                   sha256=sha(target)))
    result = dict(complete=True, root=str(root), previous=str(previous), checked_at=time.time(),
                  dataset=dataset, dataset_hardlinks_verified=len(entries) if hot_dataset is None else 0,
                  dataset_hot_copies_verified=len(entries) if hot_dataset is not None else 0,
                  hot_dataset=None if hot_dataset is None else str(hot_dataset.resolve()),
                  dataset_payload_copy_bytes=dataset.get("copied_bytes", 0), data_volume_device=volume_device,
                  python=sys.version, private_prefix=sys.prefix, torch=torch.__version__,
                  numpy=numpy.__version__, cv2=cv2.__version__, av=av.__version__,
                  recorded_packages_verified=len(expected["installed"]), package_differences=differences,
                  native_binary_sha256=digest, native_binary_imported=True,
                  native_kernel_test="actual queued production acceptance; previous qualification retained",
                  compile_caches=caches, input_assets=assets,
                  d_drive_path=str(D_DRIVE), d_drive_available_bytes=d_drive_available(),
                  e_drive_path=str(E_DRIVE), e_drive_available_bytes=e_drive_available(),
                  legacy_filesystem_growth_not_comparable_after_volume_move=True,
                  seconds=time.monotonic() - started)
    write(root / "relocation-verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"dataset", "input_assets"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--hot-dataset", type=Path)
    args = parser.parse_args()
    verify(args.root.resolve(), args.previous_root.resolve(), hot_dataset=args.hot_dataset)
