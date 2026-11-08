"""Bounded B00/shader and analytic-sheet probe. Research only; no NHT edits.

Run with the pinned NHT tool Python, PYTHONPATH=., after verified B00 download.
The public render is the control. Internal hooks expose *latent* prefix data for
diagnosis; none of the displayed prefix RGB candidates is a production contract.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from gsplat.rendering import rasterization
from nht_pipeline import render as nht
from PIL import Image, ImageDraw


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def stats(value):
    a = np.asarray(value, dtype=np.float64)
    return {
        "min": float(a.min()),
        "mean": float(a.mean()),
        "q95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def panel(path, panels):
    canvas = Image.new("RGB", (256 * len(panels), 170), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, value) in enumerate(panels):
        a = np.asarray(value)
        if a.shape[-1] == 1:
            a = np.repeat(a, 3, axis=-1)
        image = Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255 + 0.5))
        canvas.paste(image.resize((256, 144)), (256 * index, 26))
        draw.text((256 * index + 4, 4), label, fill="black")
    canvas.save(path)


def known_sheets(output):
    """Central ray hits thin Gaussian sheets exactly at their means (alpha known)."""
    cases = {
        "opaque_before": ([2.0], [0.999], [[1, 0, 0]], [0.999, 0.001, 0]),
        "opaque_behind": ([8.0], [0.999], [[0, 0, 1]], [0, 1, 0]),
        "net_before": (
            [2.0, 8.0],
            [0.25, 0.999],
            [[1, 0, 0], [0, 0, 1]],
            [0.25, 0.75, 0],
        ),
        "net_behind": ([6.0, 8.0], [0.25, 0.999], [[1, 0, 0], [0, 0, 1]], [0, 1, 0]),
    }
    records = {}
    for name, (depths, alphas, colors, expected) in cases.items():
        count = len(depths)
        kwargs = dict(
            means=torch.tensor(
                [[0, 0, d] for d in depths], device="cuda", dtype=torch.float32
            ),
            quats=torch.tensor(
                [[1, 0, 0, 0]] * count, device="cuda", dtype=torch.float32
            ),
            scales=torch.tensor(
                [[1, 1, 0.02]] * count, device="cuda", dtype=torch.float32
            ),
            opacities=torch.tensor(alphas, device="cuda", dtype=torch.float32),
            colors=torch.tensor(colors, device="cuda", dtype=torch.float32),
            viewmats=torch.eye(4, device="cuda")[None],
            Ks=torch.tensor(
                [[[20, 0, 2.5], [0, 20, 2.5], [0, 0, 1]]],
                device="cuda",
                dtype=torch.float32,
            ),
            width=5,
            height=5,
            packed=False,
            with_eval3d=True,
            with_ut=True,
            render_mode="RGB+ED",
            near_plane=0.01,
        )
        with torch.inference_mode():
            full, full_a, _ = rasterization(**kwargs, far_plane=100)
            prefix, prefix_a, _ = rasterization(**kwargs, far_plane=4)
        f = full[0, 2, 2].cpu().numpy()
        p = prefix[0, 2, 2, :3].cpu().numpy()
        a = float(prefix_a[0, 2, 2, 0])
        composite = p + (1 - a) * np.array([0, 1, 0])
        error = float(np.max(np.abs(composite - expected)))
        if error > 2e-5:
            raise AssertionError(f"Known-sheet oracle failed: {name}: {composite}")
        records[name] = {
            "z": depths,
            "alpha": alphas,
            "straight_rgb": colors,
            "full_rgb": f[:3].tolist(),
            "full_alpha": float(full_a[0, 2, 2, 0]),
            "expected_z": float(f[3]),
            "prefix_rgb": p.tolist(),
            "prefix_transmittance": 1 - a,
            "composite": composite.tolist(),
            "hand_expected": expected,
            "max_error": error,
            "ed_compare": [0, 1, 0] if f[3] > 4 else f[:3].tolist(),
        }
    write_json(output / "known-sheets.json", records)
    return records


def look_at(eye, target):
    forward = np.asarray(target) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = eye
    return pose


def cameras(asset_root, court_id):
    alignment = json.loads((asset_root / "alignment/alignment.json").read_text())
    if alignment["schema"] != "semantic_multi_court_alignment_v2":
        raise ValueError("Unexpected alignment schema")
    adapter = alignment["metric_scene_adapter"]
    scale = float(adapter["nht_scene_units_per_metre"])
    similarity = np.asarray(adapter["nht_scene_from_metric_scene"]).reshape(4, 4)
    court = next(
        c for c in alignment["layout"]["courts"] if c["court_instance_id"] == court_id
    )
    scene_from_court = np.asarray(court["scene_from_court"]).reshape(4, 4)
    transform = similarity @ scene_from_court
    requests, descriptions = [], []
    for index, eye in enumerate(([0, -8, 1.65], [0, -3, 1.65], [3, -8, 1.65])):
        pose = transform @ look_at(np.asarray(eye, dtype=float), [0, 4, 0.4])
        pose[:3, :3] /= scale  # camera pose remains rigid: scale only its translation
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-8):
            raise ValueError("Camera rotation is not orthonormal")
        k = [[180, 0, 128], [0, 180, 72], [0, 0, 1]]
        requests.append(
            dict(
                camera_id=f"court-eye-{index}",
                width=256,
                height=144,
                intrinsics=dict(
                    model="PINHOLE",
                    distortion_model="NONE",
                    params=[180, 180, 128, 72],
                    matrix=k,
                ),
                camera_to_scene=pose.tolist(),
            )
        )
        descriptions.append(
            dict(
                camera_id=f"court-eye-{index}",
                eye_court_m=eye,
                target_court_m=[0, 4, 0.4],
            )
        )
    return requests, scale, descriptions


def render_capture(renderer, camera):
    captured = []

    def capture(*args, **kwargs):
        value = rasterization(*args, **kwargs)
        captured.append(value[0])
        return value

    with patch("gsplat.rendering.rasterization", side_effect=capture):
        result, seconds = timed(lambda: renderer.render(camera))
    if len(captured) != 1:
        raise RuntimeError("Unexpected rasterization call count")
    return result, captured[0], seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--court", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    requests, scale, descriptions = cameras(args.asset_root, args.court)
    request_path = output / "camera-request.json"
    write_json(request_path, dict(schema="nht_render_request_v1", cameras=requests))
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    report = dict(
        schema_version=1,
        court=args.court,
        nht_units_per_metre=scale,
        cameras=descriptions,
        resolution=[256, 144],
        gpu=torch.cuda.get_device_name(),
        python=platform.python_version(),
        cuda=torch.version.cuda,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        packages={
            name: importlib.metadata.version(name)
            for name in ("torch", "gsplat", "numpy", "tinycudann")
        },
        known_sheets=known_sheets(output),
        views=[],
    )
    scene_path = args.asset_root / "reconstruction/export/scene.json"
    public, seconds = timed(
        lambda: nht.render_scene(
            scene_path,
            output / "public",
            request_path=request_path,
            batch_size=1,
            without_source_images=True,
        )
    )
    report["public_seconds_including_load"] = seconds
    validated = nht.load_validated_scene_export(scene_path, without_source_images=True)
    renderer, seconds = timed(
        lambda: nht.ResidentSceneRenderer(
            validated.checkpoint_path, validated.runtime_config
        )
    )
    report["resident_load_seconds"] = seconds
    report["runtime_config"] = validated.runtime_config
    report["gaussians"] = len(renderer.splats["means"])
    report["encoded_dim"] = renderer.shader.encoded_dim
    config = dict(renderer.config)
    foreground = np.zeros((144, 256, 3), dtype=np.float32)
    foreground[..., 1] = 1
    for camera, public_record in zip(requests, public["renders"], strict=True):
        root = output / camera["camera_id"]
        root.mkdir()
        (rgb, alpha, depth), latent, seconds = render_capture(renderer, camera)
        control = np.load(output / "public" / public_record["rgb"])
        if not np.allclose(rgb, control, atol=2e-5, rtol=0):
            raise AssertionError("Instrumented control differs from public RGB")
        repeated = [timed(partial(renderer.render, camera))[1] for _ in range(3)]
        item = dict(
            camera_id=camera["camera_id"],
            full_alpha=stats(alpha),
            full_depth_m=stats(depth / scale),
            capture_seconds=seconds,
            steady_seconds=repeated,
            prefix=[],
        )
        with torch.inference_mode():
            zero = latent.clone()
            zero[..., : renderer.shader.encoded_dim] = 0
            zero_rgb = renderer.shader(zero)[0][0].cpu().numpy()
            item["zero_features_decoded_rgb"] = stats(zero_rgb)
            item["affinity_error"] = []
            for weight in (0.25, 0.5, 0.75):
                scaled = latent.clone()
                scaled[..., : renderer.shader.encoded_dim] *= weight
                decoded = renderer.shader(scaled)[0][0].cpu().numpy()
                error = np.abs(decoded - (weight * rgb + (1 - weight) * zero_rgb))
                item["affinity_error"].append(dict(weight=weight, error=stats(error)))
        for cutoff_m in (2.0, 6.0, 10.0):
            renderer.config = {**config, "far_plane": cutoff_m * scale}
            (p_rgb, p_alpha, _), p_latent, prefix_seconds = render_capture(
                renderer, camera
            )
            t = 1 - p_alpha
            # Three diagnostic candidates, intentionally not claimed as physical RGB.
            raw_candidate = p_rgb + t * foreground
            weighted_candidate = p_alpha * p_rgb + t * foreground
            with torch.inference_mode():
                normalized = p_latent.clone()
                normalized[..., : renderer.shader.encoded_dim] /= (
                    torch.from_numpy(p_alpha).cuda()[None].clamp_min(1e-6)
                )
                n_rgb = renderer.shader(normalized)[0][0].cpu().numpy()
            normalized_candidate = p_alpha * n_rgb + t * foreground
            ed_candidate = np.where(
                (alpha > 0) & (depth > 0) & (depth < cutoff_m * scale), rgb, foreground
            )
            np.savez_compressed(
                root / f"prefix-{cutoff_m:g}m.npz",
                alpha=p_alpha,
                decoded_rgb=p_rgb,
                normalized_decoded_rgb=n_rgb,
                raw_candidate=raw_candidate,
                weighted_candidate=weighted_candidate,
                normalized_candidate=normalized_candidate,
            )
            panel(
                root / f"prefix-{cutoff_m:g}m.png",
                [
                    ("Full learned RGB", rgb),
                    ("Prefix alpha (mean-z gate)", p_alpha),
                    ("ED comparison (rejected)", ed_candidate),
                    ("Weighted prefix (unvalidated)", weighted_candidate),
                    ("Normalized prefix (unvalidated)", normalized_candidate),
                ],
            )
            # ROI is a fixed central/lower image region, not a semantic net mask.
            roi = np.zeros_like(p_alpha, dtype=bool)
            roi[55:125, 48:208] = True
            item["prefix"].append(
                dict(
                    cutoff_m=cutoff_m,
                    seconds=prefix_seconds,
                    alpha=stats(p_alpha),
                    full_opaque_prefix_transparent_pixels=int(
                        ((alpha > 0.99) & (p_alpha < 0.9)).sum()
                    ),
                    roi_mid_alpha_pixels=int(
                        ((p_alpha > 0.02) & (p_alpha < 0.9) & roi).sum()
                    ),
                    raw_candidate_out_of_range_values=int(
                        ((raw_candidate < 0) | (raw_candidate > 1)).sum()
                    ),
                    weighted_vs_normalized_error=stats(
                        np.abs(weighted_candidate - normalized_candidate)
                    ),
                )
            )
        renderer.config = config
        report["views"].append(item)
        write_json(output / "metrics.json", report)
        print(json.dumps(item), flush=True)
    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(output / "metrics.json", report)


if __name__ == "__main__":
    main()
