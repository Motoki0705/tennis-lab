"""Known opaque surfaces sampled at the same pixel centers as the NHT probe."""

import numpy as np


def body_field(camera, depth_m):
    yy, xx = np.mgrid[: camera["height"], : camera["width"]]
    q = camera["width"] / 256
    xx, yy = (xx + 0.5) / q, (yy + 0.5) / q
    mask = np.zeros_like(xx, dtype=bool)
    for cx, cy, rx, ry in (
        (128, 47, 6, 7),
        (128, 68, 9, 18),
        (117, 69, 5, 18),
        (139, 69, 5, 18),
        (121, 105, 5, 24),
        (135, 105, 5, 24),
    ):
        mask |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    return np.where(mask, depth_m, 0).astype(np.float32)[..., None], mask[..., None]


def sphere_field(camera, center_z_m, u_base, v_base, radius_m=0.0335):
    q = camera["width"] / 256
    k = np.asarray(camera["intrinsics"]["matrix"])
    yy, xx = np.mgrid[: camera["height"], : camera["width"]]
    rays = np.stack(
        (
            (xx + 0.5 - k[0, 2]) / k[0, 0],
            (yy + 0.5 - k[1, 2]) / k[1, 1],
            np.ones_like(xx),
        ),
        axis=-1,
    )
    center = np.asarray(
        [
            (u_base * q - k[0, 2]) * center_z_m / k[0, 0],
            (v_base * q - k[1, 2]) * center_z_m / k[1, 1],
            center_z_m,
        ]
    )
    dot = rays @ center
    norm = np.sum(rays * rays, axis=-1)
    discriminant = dot**2 - norm * (center @ center - radius_m**2)
    valid = discriminant >= 0
    depth = np.where(
        valid, (dot - np.sqrt(np.maximum(discriminant, 0))) / norm, 0
    ).astype(np.float32)
    return depth[..., None], valid[..., None]
