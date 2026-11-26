"""Fetch and verify the upstream PMPose assets used by the evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.checksum import dual_sha256

SOURCE_URL = "https://github.com/MiraPurkrabek/BBoxMaskPose.git"
SOURCE_COMMIT = "49a070a6f8396147323b1c4959474077dbe2ce8e"
WEIGHT_REPOSITORY = "https://huggingface.co/vrg-prague/BBoxMaskPose"
WEIGHT_REVISION = "cfc89052cf16655a52889f1f4e1b866e74a6c19c"
LICENSE = "GPL-3.0"


@dataclass(frozen=True)
class Weight:
    filename: str
    size: int
    sha256: str


WEIGHTS = {
    "pmpose-l": Weight(
        "PMPose-l-1.0.0.pth",
        1_704_836_116,
        "931b2f14486c199feee57714d6ab7e162fb904495050efa4d2142306395dc250",
    ),
    "pmpose-h": Weight(
        "PMPose-h-1.0.0.pth",
        3_259_992_404,
        "ca3e5577839e5ce22390d1d6409241f61526346c80086d3ee1990b503dd93635",
    ),
}


def verify_source(root: Path) -> None:
    if not (root / ".git").exists():
        raise FileNotFoundError(f"PMPose source checkout is missing: {root}")
    git = ["git", "-C", str(root)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_COMMIT:
        raise ValueError(
            f"PMPose source HEAD {head} differs from the pin {SOURCE_COMMIT}"
        )
    if subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError(f"PMPose source checkout has modified tracked files: {root}")


def verify_checkpoint(path: Path, weight: Weight) -> None:
    if path.stat().st_size != weight.size or dual_sha256(path) != weight.sha256:
        raise ValueError(f"PMPose checkpoint differs from {weight.filename}: {path}")


def _download(directory: Path, weight: Weight) -> None:
    target = directory / weight.filename
    if target.exists():
        verify_checkpoint(target, weight)
    else:
        with TemporaryDirectory(prefix=".download-", dir=directory) as staging:
            partial = Path(staging) / weight.filename
            url = f"{WEIGHT_REPOSITORY}/resolve/{WEIGHT_REVISION}/PMPose/{weight.filename}"
            subprocess.run(
                ["curl", "-fLsS", "--retry", "3", "-o", str(partial), url],
                check=True,
            )
            verify_checkpoint(partial, weight)
            os.link(partial, target)
    print(f"{weight.sha256}  {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model", choices=tuple(WEIGHTS))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    source = args.directory / "BBoxMaskPose"
    if not source.exists():
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        git = ["git", "-C", str(source)]
        subprocess.run(
            [*git, "fetch", "--depth=1", SOURCE_URL, SOURCE_COMMIT], check=True
        )
        subprocess.run([*git, "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    verify_source(source)
    print(f"{SOURCE_COMMIT}  {source}", flush=True)
    weights = WEIGHTS.values() if args.model is None else (WEIGHTS[args.model],)
    for weight in weights:
        _download(args.directory, weight)


if __name__ == "__main__":
    main()
