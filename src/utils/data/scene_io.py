"""Scene-directory IO shared by scene-based datasets.

The project stores each scene as a directory of ``.npy`` arrays plus
``scalars.json`` / ``meta.json`` side-cars. :func:`load_scene_payload` is the
canonical reader for that layout, used by the PLCS and BLCS datasets via
``SceneDatasetBase``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_scene_payload(scene_path: str | Path) -> dict[str, Any]:
    """Load all arrays and scalars from a scene directory.

    - ``scalars.json`` keys are merged into the payload at the top level.
    - ``meta.json`` is loaded under the ``"meta"`` key.
    - Every ``*.npy`` file is loaded with ``mmap_mode="r"`` (zero-copy) under its
      file stem.
    """
    scene_dir = Path(scene_path)
    payload: dict[str, Any] = {}

    scalars_path = scene_dir / "scalars.json"
    if scalars_path.exists():
        with open(scalars_path) as handle:
            scalars = json.load(handle)
        for key, value in scalars.items():
            payload[key] = value

    meta_path = scene_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as handle:
            payload["meta"] = json.load(handle)

    for npy_file in scene_dir.glob("*.npy"):
        payload[npy_file.stem] = np.load(npy_file, mmap_mode="r")

    return payload


__all__ = ["load_scene_payload"]
