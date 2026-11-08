"""Research reference for explicitly approximate prefix shading (no local radiance)."""

import numpy as np

MODES = ("weighted", "normalized", "closed_normalized")
ALPHA_EPS = 1e-6
CONSISTENCY_ATOL = 2e-6


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def prefix_parameters(transmittance, full_alpha, mode, *, foreground_valid=None):
    """Return decoder feature scale and appearance weight (not physical opacity)."""
    if mode not in MODES:
        raise ValueError(f"Unknown approximation mode: {mode}")
    t = np.asarray(transmittance, dtype=np.float64)
    full = np.asarray(full_alpha, dtype=np.float64)
    if t.shape != full.shape or t.shape[-1:] != (1,):
        raise ValueError("Transmission and full alpha shapes must match (...,1)")
    if not np.isfinite(t).all() or not np.isfinite(full).all():
        raise ValueError("Transmission and alpha must be finite")
    valid = np.ones_like(full, dtype=bool)
    if foreground_valid is not None:
        valid = np.asarray(foreground_valid)
        if valid.dtype != np.bool_ or valid.shape != full.shape:
            raise ValueError("Foreground valid must be bool and match alpha shape")
    if (
        np.any((t < 0) | (t > 1))
        or np.any((full < 0) | (full > 1))
        or np.any((full == 0) & valid)
    ):
        raise ValueError(
            "Transmission/alpha must be [0,1]; valid full alpha must be >0"
        )
    alpha = 1 - t
    if np.any(alpha > full + CONSISTENCY_ATOL):
        raise ValueError("Prefix alpha exceeds full alpha")
    alpha = np.minimum(alpha, full)  # only already-validated float32 roundoff
    active = (alpha > ALPHA_EPS) & valid
    scale = np.ones_like(alpha)
    if mode in ("normalized", "closed_normalized"):
        np.divide(1 if mode == "normalized" else full, alpha, out=scale, where=active)
    weight = alpha
    if mode == "closed_normalized":
        weight = np.zeros_like(alpha)
        np.divide(alpha, full, out=weight, where=valid)
    return np.where(active, scale, 0), np.where(active, weight, 0)


def composite_decoded(
    prefix_rgb,
    transmittance,
    full_alpha,
    foreground_rgb,
    background_rgb,
    foreground_valid,
    *,
    mode,
):
    """Blend display-linearized RGB, using the mode's explicitly approximate weight.

    The decoder output/foreground/background use the assumed-sRGB [0,1] encoding.
    The full-alpha closure in closed_normalized is intentional and is not claimed
    to recover an opacity-correct local RGB contribution from the trained shader.
    """
    prefix, foreground, background = [
        np.asarray(x) for x in (prefix_rgb, foreground_rgb, background_rgb)
    ]
    valid = np.asarray(foreground_valid)
    if (
        prefix.shape != foreground.shape
        or prefix.shape != background.shape
        or prefix.shape[-1:] != (3,)
    ):
        raise ValueError("RGB arrays must have identical (...,3) shape")
    if valid.dtype != np.bool_ or valid.shape != prefix.shape[:-1] + (1,):
        raise ValueError("Foreground valid must be bool (...,1)")
    for rgb in (prefix, foreground, background):
        if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError("Assumed-sRGB inputs must be finite and in [0,1]")
    _, weight = prefix_parameters(
        transmittance, full_alpha, mode, foreground_valid=valid
    )
    if weight.shape != valid.shape:
        raise ValueError("Alpha shapes do not match RGB")
    value = linear_to_srgb(
        weight * srgb_to_linear(prefix) + (1 - weight) * srgb_to_linear(foreground)
    )
    value = np.where(weight == 0, foreground, value)
    value = np.where(weight == 1, prefix, value)
    return np.where(valid, value, background).astype(np.float32)
