"""Download the pinned VidMap checkpoints without creating GPU models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache.resolve()
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import hf_hub_download
    from torch.hub import download_url_to_file

    huggingface = [
        (
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "b2359bdf726fb44ef62acca04d629dcf158053e7",
            "config.json",
            "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468",
        ),
        (
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "b2359bdf726fb44ef62acca04d629dcf158053e7",
            "model.safetensors",
            "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c",
        ),
        (
            "gberton/MegaLoc",
            "7cb9f7970d366fdf059963d04d372e503e8e9df9",
            "model.safetensors",
            "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8",
        ),
    ]
    urls = [
        (
            "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt",
            "checkpoints/romav2.0.1.pt",
            "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7",
        ),
        (
            "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth",
            "checkpoints/aliked-n16.pth",
            "5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2",
        ),
        (
            "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar",
            "geocalib/pinhole.tar",
            "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac",
        ),
    ]
    records: list[dict[str, str | int]] = []

    def check_space() -> None:
        free = shutil.disk_usage("/mnt/d").free
        if free < 112 * 1024**3:
            raise RuntimeError("Insufficient D: headroom for a checkpoint download")

    def record(path: Path, expected: str, source: str) -> None:
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"Checkpoint digest mismatch: {path}")
        records.append(
            {
                "path": str(path),
                "sha256": actual,
                "bytes": path.stat().st_size,
                "source": source,
            }
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)

    for repo, revision, filename, expected in huggingface:
        check_space()
        path = Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename))
        record(
            path,
            expected,
            f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
        )
    for url, relative, expected in urls:
        check_space()
        path = cache / "torch/hub" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            download_url_to_file(url, str(path), hash_prefix=expected, progress=False)
        record(path, expected, url)


if __name__ == "__main__":
    main()
