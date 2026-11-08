"""Compare prefix approximations with fixed NHT weights; research-only retrieval.

No submodule files are edited. Per-sample prefixes are assembled from native
scalar far-plane renders at the exact foreground depths used by this probe.
This emulates the proposed *mean-center-z* gate, not volumetric surface clipping.
"""

import argparse
import json
import runpy
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from approximation import (
    MODES,
    composite_decoded,
    linear_to_srgb,
    prefix_parameters,
    srgb_to_linear,
)
from geometry import body_field, sphere_field
from gsplat.rendering import rasterization
from nht_pipeline import render as nht
from PIL import Image, ImageDraw

OLD = Path(__file__).parents[1] / "run-player-depth-compositing-20261108"
HELPERS = runpy.run_path(str(OLD / "gpu_probe.py"))


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def error(a, b):
    difference = np.abs(np.asarray(a, dtype=np.float64) - b)
    return {"max": float(difference.max(initial=0)), "mean": float(difference.mean())}


def panels(path, images):
    w, h = images[0][1].shape[1::-1]
    canvas = Image.new("RGB", (w * len(images), h + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(images):
        image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255 + 0.5))
        canvas.paste(image, (w * index, 24))
        draw.text((w * index + 3, 5), label, fill="black")
    canvas.save(path)


class PrefixRenderer:
    def __init__(self, renderer, camera, scale):
        self.renderer, self.camera, self.scale = renderer, camera, scale
        self.e = renderer.shader.encoded_dim
        self.cache = OrderedDict()
        splats, config = renderer.splats, renderer.config
        self.config = dict(config)
        self.args = dict(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=splats["features"],
            sh_degree=None,
            viewmats=torch.linalg.inv(
                torch.tensor(
                    [camera["camera_to_scene"]], device="cuda", dtype=torch.float32
                )
            ),
            Ks=torch.tensor(
                [camera["intrinsics"]["matrix"]], device="cuda", dtype=torch.float32
            ),
            width=camera["width"],
            height=camera["height"],
            tile_size=config["tile_size"],
            packed=config["packed"],
            rasterize_mode="antialiased" if config["antialiased"] else "classic",
            render_mode="RGB",
            distributed=False,
            camera_model=config["camera_model"],
            with_ut=config["with_ut"],
            with_eval3d=config["with_eval3d"],
            near_plane=config["near_plane"],
            nht=True,
            center_ray_mode=config["deferred_opt_center_ray_encoding"],
            ray_dir_scale=renderer.shader.ray_dir_scale,
        )
        self.full, self.full_alpha = self.raw(None)
        self.background = self.decode(self.full)
        self.alpha = self.full_alpha[0].cpu().numpy()

    def raw(self, cutoff_m):
        far = (
            self.config["far_plane"]
            if cutoff_m is None
            else float(
                np.nextafter(np.float32(cutoff_m * self.scale), np.float32(-np.inf))
            )
        )
        with torch.inference_mode():
            latent, alpha, _ = rasterization(**self.args, far_plane=far)
        return latent, alpha

    def decode(self, latent):
        with torch.inference_mode():
            return self.renderer.shader(latent)[0][0].float().cpu().numpy()

    def at_depth(self, cutoff_m):
        key = float(cutoff_m)
        if key not in self.cache:
            self.cache[key] = self.raw(key)
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def field(self, depth, valid):
        start = time.perf_counter()
        latent = self.full.clone()
        latent[..., : self.e] = 0
        alpha = torch.zeros_like(self.full_alpha)
        cutoffs = np.unique(depth[valid])
        for cutoff in cutoffs:
            raw, a = self.at_depth(float(cutoff))
            mask = torch.from_numpy(valid & (depth == cutoff)).cuda()[None]
            latent[..., : self.e] = torch.where(
                mask, raw[..., : self.e], latent[..., : self.e]
            )
            alpha = torch.where(mask, a, alpha)
        torch.cuda.synchronize()
        return (
            latent,
            alpha[0].cpu().numpy(),
            {"cutoffs": len(cutoffs), "seconds": time.perf_counter() - start},
        )

    def proxy(self, latent, alpha, mode):
        scales, weight = prefix_parameters(1 - alpha, self.alpha, mode)
        data = latent.clone()
        data[..., : self.e] *= torch.from_numpy(scales.astype(np.float32)).cuda()[None]
        return self.decode(data), weight


def downsample(image, q):
    if q == 1:
        return image
    h, w = image.shape[:2]
    return linear_to_srgb(
        srgb_to_linear(image).reshape(h // q, q, w // q, q, 3).mean(axis=(1, 3))
    ).astype(np.float32)


def inspect_case(p, depth, valid, foreground, *, name, q, output, save=False):
    latent, alpha, retrieval = p.field(depth, valid)
    rgb_fg = np.broadcast_to(foreground, p.background.shape).astype(np.float32)
    full = p.alpha
    record = {
        "name": name,
        "samples": int(valid.sum()),
        "coverage_pixels": float(valid.sum() / q**2),
        "prefix_retrieval": retrieval,
        "full_alpha_min_on_foreground": float(full[valid].min()) if valid.any() else 1,
        "prefix_alpha_mean": float(alpha[valid].mean()) if valid.any() else 0,
        "monotonic_violation": float(np.maximum(alpha - full, 0).max()),
        "harmonic_envelope_violation": float(
            np.maximum(np.abs(latent[0, ..., : p.e].cpu().numpy()) - alpha, 0).max()
        ),
        "modes": {},
    }
    images = [("Original NHT", downsample(p.background, q))]
    for mode in MODES:
        proxy, weight = p.proxy(latent, alpha, mode)
        actual = composite_decoded(
            proxy, 1 - alpha, full, rgb_fg, p.background, valid, mode=mode
        )
        black = composite_decoded(
            proxy,
            1 - alpha,
            full,
            np.zeros_like(rgb_fg),
            p.background,
            valid,
            mode=mode,
        )
        expected_response = np.where(valid, (1 - weight) * srgb_to_linear(rgb_fg), 0)
        response = srgb_to_linear(actual) - srgb_to_linear(black)
        outside = np.broadcast_to(~valid, actual.shape)
        record["modes"][mode] = {
            "outside_mask_max_error": float(
                np.abs(actual - p.background)[outside].max(initial=0)
            ),
            "response_max_error": float(np.abs(response - expected_response).max()),
            "finite": bool(np.isfinite(actual).all()),
            "range_min": float(actual.min()),
            "range_max": float(actual.max()),
            "foreground_transmission_mean": float((1 - weight)[valid].mean())
            if valid.any()
            else 1,
            "image_vs_original_mean_abs_delta": float(
                np.abs(actual - p.background).mean()
            ),
            "ball_green_response_sum_linear": float(response[..., 1].sum() / q**2),
        }
        images.append((mode, downsample(actual, q)))
        if save:
            np.savez_compressed(
                output / f"{name}-{mode}.npz",
                image=actual,
                prefix_rgb=proxy,
                prefix_features=latent[0, ..., : p.e].cpu().numpy(),
                prefix_alpha=alpha,
                full_alpha=full,
                depth_m=depth,
                foreground_valid=valid,
            )
    if save:
        panels(output / f"{name}.png", images)
    if "ball-" in name or "sphere-" in name:
        crops = []
        for label, image in images:
            crop = image[62:78, 120:136]
            crops.append((label, np.repeat(np.repeat(crop, 6, axis=0), 6, axis=1)))
        panels(output / f"{name}-crop.png", crops)
    return record


def endpoints(p):
    shape = p.background.shape
    fg = np.zeros(shape, np.float32)
    fg[..., 1] = 1
    on, off = np.ones(shape[:-1] + (1,), bool), np.zeros(shape[:-1] + (1,), bool)
    result = {}
    empty = p.full.clone()
    empty[..., : p.e] = 0
    for mode in MODES:
        proxy, _ = p.proxy(p.full, p.alpha, mode)
        all_image = composite_decoded(
            proxy, 1 - p.alpha, p.alpha, fg, p.background, on, mode=mode
        )
        empty_proxy, _ = p.proxy(empty, np.zeros_like(p.alpha), mode)
        empty_image = composite_decoded(
            empty_proxy, np.ones_like(p.alpha), p.alpha, fg, p.background, on, mode=mode
        )
        none_image = composite_decoded(
            empty_proxy,
            np.ones_like(p.alpha),
            p.alpha,
            fg,
            p.background,
            off,
            mode=mode,
        )
        result[mode] = {
            "no_foreground": error(none_image, p.background),
            "empty_prefix": error(empty_image, fg),
            "full_prefix": error(all_image, p.background),
        }
    # Uniform attenuation of a fixed real feature average must not change a
    # normalized variant's straight proxy color; this is a stability objective.
    for mode in MODES:
        decoded = []
        for alpha in (0.005, 0.02, 0.1, 0.5, 0.99):
            a = np.full_like(p.alpha, alpha) * p.alpha
            data = p.full.clone()
            data[..., : p.e] *= alpha
            decoded.append(p.proxy(data, a, mode)[0])
        result[mode]["opacity_rescaling_straight_rgb_error"] = max(
            error(value, decoded[-1])["max"] for value in decoded[:-1]
        )
    return result


def large_gaussian(renderer, output):
    # One tilted, broad Gaussian straddles z=4.2; an opaque sheet lies at z=10.
    means = torch.tensor(
        [[0.6, 0, 4.2], [0, 0, 10]], device="cuda", dtype=torch.float32
    )
    angle = np.pi / 4
    quats = torch.tensor(
        [[np.cos(angle / 2), 0, np.sin(angle / 2), 0], [1, 0, 0, 0]],
        device="cuda",
        dtype=torch.float32,
    )
    scales = torch.tensor(
        [[0.3, 0.5, 1.8], [20, 20, 0.02]], device="cuda", dtype=torch.float32
    )
    features = (
        renderer.splats["features"][:2]
        .reshape(2, 4, -1)[:, :1]
        .repeat(1, 4, 1)
        .reshape(2, -1)
    )
    arguments = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=torch.tensor([0.8, 0.999], device="cuda"),
        colors=features,
        viewmats=torch.eye(4, device="cuda")[None],
        Ks=torch.tensor(
            [[[55, 0, 32], [0, 55, 18], [0, 0, 1]]], device="cuda", dtype=torch.float32
        ),
        width=64,
        height=36,
        near_plane=0.01,
        packed=False,
        with_ut=True,
        with_eval3d=True,
        nht=True,
        render_mode="RGB",
        ray_dir_scale=renderer.shader.ray_dir_scale,
    )
    with torch.inference_mode():
        full, a, _ = rasterization(**arguments, far_plane=100)
        rgb = renderer.shader(full)[0][0].cpu().numpy()
        full_a = a[0].cpu().numpy()
        outputs = []
        alphas = []
        for cutoff in (4.199, 4.201):
            prefix, pa, _ = rasterization(**arguments, far_plane=cutoff)
            alpha = pa[0].cpu().numpy()
            alphas.append(alpha)
            rows = [
                (
                    "Original learned shader",
                    np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1),
                )
            ]
            for mode in MODES:
                scaling, _ = prefix_parameters(1 - alpha, full_a, mode)
                value = prefix.clone()
                value[..., : renderer.shader.encoded_dim] *= torch.from_numpy(
                    scaling.astype(np.float32)
                ).cuda()[None]
                proxy = renderer.shader(value)[0][0].cpu().numpy()
                fg = np.zeros_like(rgb)
                fg[..., 1] = 1
                image = composite_decoded(
                    proxy,
                    1 - alpha,
                    full_a,
                    fg,
                    rgb,
                    np.ones_like(alpha, bool),
                    mode=mode,
                )
                rows.append((mode, np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)))
                if mode == "closed_normalized":
                    outputs.append(image)
            panels(output / f"large-gaussian-{cutoff}.png", rows)
    # Independent analytic Mahalanobis ray-peak depth, compared to mean-z gate.
    yy, xx = np.mgrid[:36, :64]
    rays = np.stack(((xx + 0.5 - 32) / 55, (yy + 0.5 - 18) / 55, np.ones_like(xx)), -1)
    rot = np.asarray(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    precision = rot @ np.diag(1 / np.array([0.3, 0.5, 1.8]) ** 2) @ rot.T
    mu = np.array([0.6, 0, 4.2])
    peak_z = (rays @ precision @ mu) / np.einsum(
        "...i,ij,...j->...", rays, precision, rays
    )
    active = alphas[1][..., 0] > 0.02
    return {
        "mean_z": 4.2,
        "cutoffs_m": [4.199, 4.201],
        "max_prefix_alpha_jump": float(np.abs(alphas[1] - alphas[0]).max()),
        "closed_rgb_jump_max": error(outputs[0], outputs[1])["max"],
        "peak_before_4m_but_mean_behind_pixels": int(((peak_z < 4) & active).sum()),
        "relevant_pixels": int(active.sum()),
        "known_large_gaussian_warning": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cameras, scale, descriptions = HELPERS["cameras"](args.asset_root, "court-000")
    dump(
        output / "camera-request.json",
        {"schema": "nht_render_request_v1", "cameras": cameras},
    )
    scene = args.asset_root / "reconstruction/export/scene.json"
    public = nht.render_scene(
        scene,
        output / "public",
        request_path=output / "camera-request.json",
        batch_size=1,
        without_source_images=True,
    )
    validated = nht.load_validated_scene_export(scene, without_source_images=True)
    renderer = nht.ResidentSceneRenderer(
        validated.checkpoint_path, validated.runtime_config
    )
    torch.cuda.reset_peak_memory_stats()
    report = {
        "schema_version": 1,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "modes": list(MODES),
        "background": "B00",
        "camera_descriptions": descriptions,
        "prefix_gate": "mean_center_z_strict_foreground_wins",
        "rgb_blend": "assumed_srgb_linearized",
        "views": [],
        "cases": [],
        "quality_is_not_ground_truth_error": True,
    }
    for index, camera in enumerate(cameras):
        p = PrefixRenderer(renderer, camera, scale)
        original = np.load(
            output / "public" / public["renders"][index]["rgb"], allow_pickle=False
        )
        record = {
            "camera": camera["camera_id"],
            "public_rgb_error": error(p.background, original),
            "endpoints": endpoints(p),
        }
        report["views"].append(record)
        near, far = (6, 10) if index != 1 else (2, 6)
        for depth, tag in ((near, "before"), (far, "after")):
            z, valid = body_field(camera, depth)
            gray = linear_to_srgb(np.array([0.18, 0.18, 0.18]))
            report["cases"].append(
                inspect_case(
                    p,
                    z,
                    valid,
                    gray,
                    name=f"eye-{index}-body-{tag}",
                    q=1,
                    output=output,
                    save=True,
                )
            )
        for depth, tag in ((near, "before"), (far, "after")):
            z, valid = sphere_field(camera, depth, 128.15, 69.83)
            report["cases"].append(
                inspect_case(
                    p,
                    z,
                    valid,
                    [0, 1, 0],
                    name=f"eye-{index}-sphere-{tag}",
                    q=1,
                    output=output,
                    save=True,
                )
            )
        # Direct sample-center sphere depths: no depth binning/ED substitution.
        for step, u in enumerate(np.linspace(125.1, 130.9, 8)):
            z, valid = sphere_field(camera, far, u, 69.83)
            report["cases"].append(
                inspect_case(
                    p,
                    z,
                    valid,
                    [0, 1, 0],
                    name=f"eye-{index}-ball-{step}",
                    q=1,
                    output=output,
                    save=step == 3,
                )
            )
        dump(output / "metrics.json", report)
        print(json.dumps(record), flush=True)
        del p
    # Repeat two cameras with a common 2x2 sampling grid, for thin balls/edges.
    for index in (0, 1):
        camera = json.loads(json.dumps(cameras[index]))
        camera["width"] *= 2
        camera["height"] *= 2
        k = np.asarray(camera["intrinsics"]["matrix"], float)
        k[:2] *= 2
        camera["intrinsics"]["matrix"] = k.tolist()
        camera["intrinsics"]["params"] = [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]
        p = PrefixRenderer(renderer, camera, scale)
        depth = 10 if index == 0 else 6
        for step, u in enumerate(np.linspace(125.1, 130.9, 8)):
            z, valid = sphere_field(camera, depth, u, 69.83)
            report["cases"].append(
                inspect_case(
                    p,
                    z,
                    valid,
                    [0, 1, 0],
                    name=f"ssaa2-eye-{index}-ball-{step}",
                    q=2,
                    output=output,
                    save=step == 3,
                )
            )
        z, valid = body_field(camera, depth)
        report["cases"].append(
            inspect_case(
                p,
                z,
                valid,
                linear_to_srgb(np.array([0.18, 0.18, 0.18])),
                name=f"ssaa2-eye-{index}-body",
                q=2,
                output=output,
                save=True,
            )
        )
        del p
    report["large_gaussian"] = large_gaussian(renderer, output)
    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    dump(output / "metrics.json", report)
    print(
        json.dumps(
            {"cases": len(report["cases"]), "large_gaussian": report["large_gaussian"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
