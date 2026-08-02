"""Render a diagnostic preview from a PLCS control comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equal_axes(axis: object, points: np.ndarray) -> None:
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = float(np.ptp(points, axis=0).max() / 2.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)  # type: ignore[attr-defined]
    axis.set_ylim(center[1] - radius, center[1] + radius)  # type: ignore[attr-defined]
    axis.set_zlim(center[2] - radius, center[2] + radius)  # type: ignore[attr-defined]


def render_preview(*, artifact: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preview: {output}")
    manifest = json.loads((artifact / "manifest.json").read_text())
    if manifest.get("schema") != "plcs_avatar_control_probe_v1":
        raise ValueError("Unsupported control-probe schema.")
    for filename, expected in manifest["files"].items():
        actual = _sha256(artifact / filename)
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch: {filename}")

    target = np.load(artifact / "target_mesh_attachments_m.npy", allow_pickle=False)
    gaussianavatar = np.load(
        artifact / "gaussianavatar_query_lbs_m.npy",
        allow_pickle=False,
    )
    hugs = np.load(artifact / "hugs_topk_lbs_m.npy", allow_pickle=False)
    error = np.linalg.norm(gaussianavatar - target, axis=-1)
    frame = int(np.unravel_index(error.argmax(), error.shape)[0])

    figure = plt.figure(figsize=(12, 4.2), constrained_layout=True)
    panels = (
        ("SMPL-X mesh attachments", target[frame], "#111111"),
        ("GaussianAvatar-style fixed LBS", gaussianavatar[frame], "#e76f51"),
        ("HUGS-style top-k LBS", hugs[frame], "#277da1"),
    )
    for index, (title, points, color) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.scatter(
            target[frame, :, 0],
            target[frame, :, 1],
            target[frame, :, 2],
            c="#b8b8b8",
            depthshade=False,
            s=4,
            alpha=0.55,
        )
        if index > 1:
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=color,
                depthshade=False,
                s=5,
                alpha=0.75,
            )
        _equal_axes(axis, target[frame])
        axis.view_init(elev=12, azim=-72)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
    figure.suptitle(
        "PLCS geometry-control screen — frame "
        f"{frame}; colored predictions overlay gray mesh attachments",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=160,
        metadata={"Software": "tennis-lab PLCS control probe"},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_preview(artifact=args.artifact, output=args.output)
    print(f"preview={args.output}")
    print(f"sha256={_sha256(args.output)}")
