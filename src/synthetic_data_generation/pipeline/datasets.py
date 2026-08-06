"""Court, BLCS and PLCS sample generation from scene/alignment contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .process import run_process
from .scene import StandardScene
from .stages import Target

COURT_KEYPOINTS = np.asarray(
    [
        [-5.485, 0.0, -11.885],
        [5.485, 0.0, -11.885],
        [-5.485, 0.0, 11.885],
        [5.485, 0.0, 11.885],
        [-4.115, 0.0, -11.885],
        [4.115, 0.0, -11.885],
        [-4.115, 0.0, 11.885],
        [4.115, 0.0, 11.885],
        [-4.115, 0.0, -6.40],
        [4.115, 0.0, -6.40],
        [-4.115, 0.0, 6.40],
        [4.115, 0.0, 6.40],
        [0.0, 0.0, -6.40],
        [0.0, 0.0, 6.40],
    ],
    dtype=np.float64,
)


def _load_render_array(
    render_root: Path,
    relative_path: object,
    *,
    expected_shape: tuple[int, ...],
    value_name: str,
) -> NDArray[np.float32]:
    if not isinstance(relative_path, str):
        raise ValueError(f"Renderer omitted the {value_name} array path")
    path = (render_root / relative_path).resolve()
    try:
        path.relative_to(render_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Renderer {value_name} path escapes its output root"
        ) from error
    if not path.is_file():
        raise ValueError(f"Renderer did not publish {value_name}: {relative_path}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32:
        raise ValueError(f"Renderer {value_name} must use float32")
    if value.shape != expected_shape:
        raise ValueError(
            f"Renderer {value_name} shape {value.shape} != {expected_shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"Renderer {value_name} contains non-finite values")
    return value


def _validate_renders(
    render_root: Path,
    render_manifest: Mapping[str, Any],
    cameras: Sequence[Mapping[str, Any]],
    scene_id: str,
) -> None:
    if render_manifest.get("schema") != "nht_render_result_v1":
        raise ValueError("Renderer did not publish nht_render_result_v1")
    if render_manifest.get("scene_schema") != "nht_standard_scene_v1":
        raise ValueError("Renderer did not preserve the standard scene schema")
    if render_manifest.get("scene_id") != scene_id:
        raise ValueError("Renderer result belongs to a different scene")
    records = render_manifest.get("renders")
    if not isinstance(records, list):
        raise ValueError("Renderer manifest must contain a renders list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for value in records:
        if not isinstance(value, Mapping) or not isinstance(
            value.get("camera_id"), str
        ):
            raise ValueError("Renderer manifest contains an invalid render record")
        identifier = value["camera_id"]
        if identifier in by_id:
            raise ValueError(f"Renderer returned duplicate camera_id {identifier!r}")
        by_id[identifier] = value
    expected_ids = {str(camera["camera_id"]) for camera in cameras}
    if set(by_id) != expected_ids:
        raise ValueError("Renderer camera IDs do not match the requested cameras")

    for camera in cameras:
        identifier = str(camera["camera_id"])
        record = by_id[identifier]
        height = int(camera["height"])
        width = int(camera["width"])
        if record.get("height") != height or record.get("width") != width:
            raise ValueError(
                f"Renderer resolution does not match camera {identifier!r}"
            )
        rgb = _load_render_array(
            render_root,
            record.get("rgb"),
            expected_shape=(height, width, 3),
            value_name=f"{identifier} RGB",
        )
        alpha = _load_render_array(
            render_root,
            record.get("alpha"),
            expected_shape=(height, width, 1),
            value_name=f"{identifier} alpha",
        )
        depth = _load_render_array(
            render_root,
            record.get("depth"),
            expected_shape=(height, width, 1),
            value_name=f"{identifier} depth",
        )
        if rgb.min() < 0 or rgb.max() > 1:
            raise ValueError(f"Renderer RGB is outside [0, 1] for {identifier!r}")
        if alpha.min() < 0 or alpha.max() > 1:
            raise ValueError(f"Renderer alpha is outside [0, 1] for {identifier!r}")
        if depth.min() < 0:
            raise ValueError(f"Renderer depth is negative for {identifier!r}")


def _transform(
    matrix: NDArray[np.float64], points: NDArray[np.float64]
) -> NDArray[np.float64]:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (matrix @ homogeneous.T).T[:, :3]


def _project(
    camera: Mapping[str, Any], scene_points: NDArray[np.float64]
) -> list[list[float] | None]:
    camera_to_scene = np.asarray(camera["camera_to_scene"], dtype=np.float64)
    scene_to_camera = np.asarray(np.linalg.inv(camera_to_scene), dtype=np.float64)
    camera_points = _transform(scene_to_camera, scene_points)
    intrinsic = np.asarray(camera["intrinsics"]["matrix"], dtype=np.float64)
    projected = (intrinsic @ camera_points.T).T
    result: list[list[float] | None] = []
    for value, point in zip(projected, camera_points, strict=True):
        if point[2] <= 1.0e-6:
            result.append(None)
        else:
            result.append([float(value[0] / value[2]), float(value[1] / value[2])])
    return result


def _visible_count(camera: Mapping[str, Any], scene_points: NDArray[np.float64]) -> int:
    width = int(camera["width"])
    height = int(camera["height"])
    return sum(
        point is not None and 0.0 <= point[0] < width and 0.0 <= point[1] < height
        for point in _project(camera, scene_points)
    )


def _selected_cameras(
    scene: StandardScene,
    sample_count: int,
    scene_from_court: NDArray[np.float64],
    target: Target,
    *,
    require_visibility: bool,
) -> tuple[dict[str, Any], ...]:
    court_points = _transform(scene_from_court, COURT_KEYPOINTS)
    count = min(sample_count, len(scene.cameras))
    frame_indices = [
        int(camera.get("source_frame_index", 0)) for camera in scene.cameras
    ]
    desired_frames = np.linspace(min(frame_indices), max(frame_indices), count)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for index, desired_frame in enumerate(desired_frames):
        target_points = _selection_points(target, index, count)
        target_scene = _transform(scene_from_court, target_points)
        ranked: list[tuple[int, float, int, int, str, dict[str, Any]]] = []
        for camera in scene.cameras:
            identifier = str(camera["camera_id"])
            if identifier in used:
                continue
            target_visible = _visible_count(camera, target_scene)
            court_visible = _visible_count(camera, court_points)
            frame_index = int(camera.get("source_frame_index", 0))
            ranked.append(
                (
                    target_visible,
                    abs(frame_index - desired_frame),
                    court_visible,
                    frame_index,
                    identifier,
                    camera,
                )
            )
        fully_visible = [item for item in ranked if item[0] == len(target_points)]
        if not fully_visible:
            if require_visibility:
                raise ValueError(
                    f"No unused exported camera fully sees {target.value} sample {index}"
                )
            fully_visible = ranked
        choice = min(
            fully_visible,
            key=lambda item: (item[1], -item[2], item[3], item[4]),
        )
        selected.append(choice[5])
        used.add(choice[4])
    return tuple(selected)


def _selection_points(target: Target, index: int, count: int) -> NDArray[np.float64]:
    if target is Target.COURT:
        return COURT_KEYPOINTS
    phase = index / max(count - 1, 1)
    if target is Target.BLCS:
        return np.asarray(
            [
                [
                    -3.5 + 7.0 * phase,
                    1.0 + 2.0 * np.sin(np.pi * phase),
                    -8.0 + 16.0 * phase,
                ]
            ],
            dtype=np.float64,
        )
    feet = np.asarray(
        [
            [-2.0 + phase, 0.0, -8.0 + 2.0 * phase],
            [2.0 - phase, 0.0, 8.0 - 2.0 * phase],
        ],
        dtype=np.float64,
    )
    heads = feet.copy()
    heads[:, 1] = 1.8
    return np.vstack([feet, heads])


def _domain_labels(
    target: Target,
    camera: Mapping[str, Any],
    scene_from_court: NDArray[np.float64],
    index: int,
    count: int,
) -> dict[str, Any]:
    if target is Target.COURT:
        points = COURT_KEYPOINTS
        return {
            "court_keypoints_court_m": points.tolist(),
            "court_keypoints_image_px": _project(
                camera, _transform(scene_from_court, points)
            ),
        }
    phase = index / max(count - 1, 1)
    if target is Target.BLCS:
        point = np.asarray(
            [
                [
                    -3.5 + 7.0 * phase,
                    1.0 + 2.0 * np.sin(np.pi * phase),
                    -8.0 + 16.0 * phase,
                ]
            ],
            dtype=np.float64,
        )
        return {
            "ball_position_court_m": point[0].tolist(),
            "ball_position_image_px": _project(
                camera, _transform(scene_from_court, point)
            )[0],
        }
    players = np.asarray(
        [
            [-2.0 + phase, 0.0, -8.0 + 2.0 * phase],
            [2.0 - phase, 0.0, 8.0 - 2.0 * phase],
        ],
        dtype=np.float64,
    )
    return {
        "players": [
            {
                "player_id": player_id,
                "position_court_m": point.tolist(),
                "yaw_radians": 0.0 if player_id == 0 else float(np.pi),
                "position_image_px": projected,
            }
            for player_id, (point, projected) in enumerate(
                zip(
                    players,
                    _project(camera, _transform(scene_from_court, players)),
                    strict=True,
                )
            )
        ]
    }


def _paint_disk(
    image: NDArray[np.float32],
    mask: NDArray[np.uint8],
    center: list[float] | None,
    radius: int,
    color: tuple[float, float, float],
    instance_id: int,
) -> None:
    if center is None:
        return
    height, width = mask.shape
    x, y = center
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(width, int(np.ceil(x + radius + 1)))
    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(height, int(np.ceil(y + radius + 1)))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    selected = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    image[y0:y1, x0:x1][selected] = color
    mask[y0:y1, x0:x1][selected] = instance_id


def _paint_player(
    image: NDArray[np.float32],
    mask: NDArray[np.uint8],
    foot: list[float] | None,
    head: list[float] | None,
    width_px: int,
    color: tuple[float, float, float],
    instance_id: int,
) -> None:
    if foot is None or head is None:
        return
    height, width = mask.shape
    x0, y0 = foot
    x1, y1 = head
    low_x = max(0, int(np.floor(min(x0, x1) - width_px)))
    high_x = min(width, int(np.ceil(max(x0, x1) + width_px + 1)))
    low_y = max(0, int(np.floor(min(y0, y1) - width_px)))
    high_y = min(height, int(np.ceil(max(y0, y1) + width_px + 1)))
    if low_x >= high_x or low_y >= high_y:
        return
    yy, xx = np.mgrid[low_y:high_y, low_x:high_x]
    dx = x1 - x0
    dy = y1 - y0
    denominator = max(dx * dx + dy * dy, 1.0e-6)
    phase = np.clip(((xx - x0) * dx + (yy - y0) * dy) / denominator, 0.0, 1.0)
    distance = np.hypot(xx - (x0 + phase * dx), yy - (y0 + phase * dy))
    selected = distance <= width_px
    image[low_y:high_y, low_x:high_x][selected] = color
    mask[low_y:high_y, low_x:high_x][selected] = instance_id


def _composite_domain_sample(
    target: Target,
    camera: Mapping[str, Any],
    scene_from_court: NDArray[np.float64],
    label: Mapping[str, Any],
    render_root: Path,
) -> tuple[str | None, int]:
    if target is Target.COURT:
        return None, 0
    camera_root = render_root / str(camera["camera_id"])
    rgb_path = camera_root / "rgb.npy"
    alpha_path = camera_root / "alpha.npy"
    depth_path = camera_root / "depth.npy"
    rgb = np.load(rgb_path, allow_pickle=False)
    alpha = np.load(alpha_path, allow_pickle=False)
    depth = np.load(depth_path, allow_pickle=False)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    fx = float(camera["intrinsics"]["matrix"][0][0])

    if target is Target.BLCS:
        point = np.asarray([label["ball_position_court_m"]], dtype=np.float64)
        scene_point = _transform(scene_from_court, point)
        camera_point = _transform(
            np.linalg.inv(np.asarray(camera["camera_to_scene"], dtype=np.float64)),
            scene_point,
        )[0]
        center = label["ball_position_image_px"]
        radius = max(2, int(round(fx * 0.0335 / max(camera_point[2], 1.0e-6))))
        _paint_disk(rgb, mask, center, radius, (0.95, 0.90, 0.10), 1)
        if center is not None:
            depth[mask == 1, 0] = float(camera_point[2])
    else:
        players = np.asarray(
            [value["position_court_m"] for value in label["players"]],
            dtype=np.float64,
        )
        heads = players.copy()
        heads[:, 1] = 1.8
        scene_feet = _transform(scene_from_court, players)
        scene_heads = _transform(scene_from_court, heads)
        projected_heads = _project(camera, scene_heads)
        camera_feet = _transform(
            np.linalg.inv(np.asarray(camera["camera_to_scene"], dtype=np.float64)),
            scene_feet,
        )
        colors = ((0.10, 0.45, 0.95), (0.95, 0.25, 0.15))
        for index, (player, head, camera_foot, color) in enumerate(
            zip(label["players"], projected_heads, camera_feet, colors, strict=True)
        ):
            width_px = max(2, int(round(fx * 0.18 / max(camera_foot[2], 1.0e-6))))
            _paint_player(
                rgb,
                mask,
                player["position_image_px"],
                head,
                width_px,
                color,
                index + 1,
            )
            if player["position_image_px"] is not None:
                depth[mask == index + 1, 0] = float(camera_foot[2])

    alpha[mask > 0, 0] = 1.0
    if not np.isfinite(rgb).all() or not np.isfinite(depth).all():
        raise ValueError("Domain composition produced non-finite values")
    np.save(rgb_path, rgb.astype(np.float32))
    np.save(alpha_path, alpha.astype(np.float32))
    np.save(depth_path, depth.astype(np.float32))
    mask_path = camera_root / "instance-mask.npy"
    np.save(mask_path, mask)
    return str(mask_path.relative_to(render_root.parent.parent)), int(
        np.count_nonzero(mask)
    )


def generate_domain_dataset(
    target: Target,
    scene: StandardScene,
    alignment_path: Path,
    output_root: Path,
    *,
    sample_count: int,
    seed: int,
    render_command: Sequence[str],
    working_directory: Path | None,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    alignment = json.loads(alignment_path.read_text())
    if alignment.get("status") != "accepted":
        raise ValueError("Dataset generation requires accepted alignment")
    scene_from_court = np.asarray(alignment["scene_from_court"], dtype=np.float64)
    production_evidence = (
        alignment["settings"]["evidence"]["mode"] == "image_achromatic"
    )
    cameras = _selected_cameras(
        scene,
        sample_count,
        scene_from_court,
        target,
        require_visibility=production_evidence,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    render_root = output_root / "samples/render"
    command = [*render_command, "--scene", str(scene.root / "scene.json")]
    for camera in cameras:
        command.extend(["--camera-id", str(camera["camera_id"])])
    command.extend(["--output", str(render_root)])
    run_process(
        command,
        working_directory=working_directory,
        environment=environment,
    )
    render_manifest = json.loads((render_root / "render.json").read_text())
    _validate_renders(
        render_root, render_manifest, cameras, str(scene.payload["scene_id"])
    )
    labels_root = output_root / "samples/labels"
    labels_root.mkdir(parents=True)
    samples = []
    for index, camera in enumerate(cameras):
        identifier = f"sample-{index:04d}"
        label = {
            "schema": f"tennis_{target.value}_sample_v1",
            "sample_id": identifier,
            "scene_id": scene.payload["scene_id"],
            "camera_id": camera["camera_id"],
            **_domain_labels(target, camera, scene_from_court, index, len(cameras)),
        }
        label_path = labels_root / f"{identifier}.json"
        label_path.write_text(json.dumps(label, indent=2) + "\n")
        mask_path, visible_instance_pixels = _composite_domain_sample(
            target, camera, scene_from_court, label, render_root
        )
        if (
            target is not Target.COURT
            and production_evidence
            and visible_instance_pixels == 0
        ):
            raise ValueError(
                f"{target.value} sample {identifier} has no visible instance pixels"
            )
        sample = {
            "sample_id": identifier,
            "camera_id": camera["camera_id"],
            "label": str(label_path.relative_to(output_root)),
            "rgb": f"samples/render/{camera['camera_id']}/rgb.npy",
            "alpha": f"samples/render/{camera['camera_id']}/alpha.npy",
            "depth": f"samples/render/{camera['camera_id']}/depth.npy",
        }
        if mask_path is not None:
            sample["instance_mask"] = mask_path
            sample["visible_instance_pixels"] = visible_instance_pixels
        samples.append(sample)
    payload = {
        "schema": "tennis_domain_dataset_v1",
        "status": "completed",
        "domain": target.value,
        "scene_id": scene.payload["scene_id"],
        "sample_count": len(samples),
        "split": {"train": len(samples), "validation": 0},
        "coordinate_space": "court metres via alignment/scene_from_court",
        "seed": seed,
        "renderer_boundary": {
            "scene": "reconstruction/export/scene.json",
            "command": list(render_command),
            "result_schema": "nht_render_result_v1",
        },
        "samples": samples,
    }
    (output_root / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
