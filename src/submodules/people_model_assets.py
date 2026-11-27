"""Official checkpoint sources and byte verification for the scene people models."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from src.utils.checksum import dual_sha256


@dataclass(frozen=True, slots=True)
class PeopleModelCheckpoint:
    relative_path: str
    source_url: str
    sha256: str

    def verify(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required checkpoint file is missing: {path}. Obtain it from {self.source_url}"
            )
        actual = dual_sha256(path)
        if actual != self.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {self.sha256}, got {actual}. "
                f"Obtain the official checkpoint from {self.source_url}"
            )


_RTMPOSE_X_FILENAME = "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth"

PEOPLE_MODEL_CHECKPOINTS = {
    "rfdetr-l": PeopleModelCheckpoint(
        "rfdetr/rf-detr-large-2026.pth",
        "https://repo.roboflow.com/rfdetr/rf-detr-large-2026.pth",
        "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38",
    ),
    "rtmpose-x": PeopleModelCheckpoint(
        f"rtmpose/{_RTMPOSE_X_FILENAME}",
        f"https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/{_RTMPOSE_X_FILENAME}",
        "7fb6e239601082a06ca8442feb7a9774b8d77a1c37911b2ad0f27c1e84be4d21",
    ),
}


def install_checkpoint(
    checkpoint: PeopleModelCheckpoint, destination: Path, *, verify_only: bool = False,
) -> None:
    """Verify an installed checkpoint or download and publish a verified file."""
    if verify_only or destination.exists():
        checkpoint.verify(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".people-model-", dir=destination.parent) as staging:
        partial = Path(staging) / destination.name
        with urlopen(checkpoint.source_url, timeout=60) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        checkpoint.verify(partial)
        try:
            os.link(partial, destination)
        except FileExistsError:
            checkpoint.verify(destination)
