"""HOG retrieval and explicit truncated-distance ECC outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .database import Array, LineDatabase, descriptor, normalize_homography


def validate_mask(mask: Array) -> Array:
    value = np.asarray(mask)
    if value.ndim != 2 or min(value.shape) < 16 or value.dtype != np.uint8:
        raise ValueError("line mask must be a 2D uint8 image with dimensions >=16")
    if not np.isin(value, [0, 255]).all():
        raise ValueError(
            "line mask must be binary 0/255; threshold probabilities explicitly"
        )
    count = np.count_nonzero(value)
    if count < 16 or count > value.size * 0.5:
        raise ValueError("line mask is blank or too dense")
    return value


def resize_query(mask: Array, size: tuple[int, int]) -> tuple[Array, Array]:
    """Resize into DB coordinates; S maps original pixel centers to DB centers.

    Anisotropic resizing is explicit (no guessed crop or aspect-ratio padding).
    OpenCV half-pixel sampling gives u' = sx*(u+0.5)-0.5.
    """
    mask = validate_mask(mask)
    sx, sy = size[0] / mask.shape[1], size[1] / mask.shape[0]
    s = np.array(
        [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64
    )
    resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST_EXACT)
    return validate_mask(resized), s


@dataclass(frozen=True)
class Match:
    index: int
    descriptor_distance: float
    initial_world_to_query: Array
    success: bool
    reason: str
    ecc: float | None
    world_to_query: Array | None
    query_to_template: Array | None


def distance_image(mask: Array, truncation: float) -> Array:
    if not np.isfinite(truncation) or truncation <= 0:
        raise ValueError("distance truncation must be positive and finite")
    distances = cv2.distanceTransform(
        255 - validate_mask(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return np.minimum(distances, truncation) / np.float32(truncation)


def refine(
    template: Array,
    query: Array,
    world_to_template: Array,
    *,
    truncation: float = 12.0,
    iterations: int = 100,
    epsilon: float = 1e-6,
) -> tuple[Array | None, Array | None, float | None, str]:
    """Return (world→query, query→template, ECC, status); failure has no H.

    findTransformECC(templateImage=query, inputImage=template) estimates the
    sampling map query→template. Therefore H_query = inverse(W) @ H_template.
    """
    if template.shape != query.shape:
        raise ValueError("refinement masks must have identical shapes")
    if (
        type(iterations) is not int
        or iterations < 1
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("invalid ECC termination criteria")
    h = normalize_homography(world_to_template)
    q_dist, t_dist = (
        distance_image(query, truncation),
        distance_image(template, truncation),
    )
    try:
        score, warp = cv2.findTransformECC(
            q_dist,
            t_dist,
            np.eye(3, dtype=np.float32),
            cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, epsilon),
            None,
            5,
        )
    except cv2.error as error:
        return None, None, None, f"ecc_failed: {error}"
    try:
        warp = normalize_homography(warp)
        result = normalize_homography(np.linalg.inv(warp) @ h)
        # Reject horizons/folds within the query image, even if ECC returned.
        height, width = query.shape
        corners = np.array(
            [
                [0, 0, 1],
                [width - 1, 0, 1],
                [0, height - 1, 1],
                [width - 1, height - 1, 1],
            ]
        )
        if (
            np.any((corners @ warp.T)[:, 2] <= 0)
            or np.linalg.det(warp) <= 0
            or not np.isfinite(score)
        ):
            raise ValueError("ECC returned invalid image mapping")
    except (ValueError, np.linalg.LinAlgError) as error:
        return None, None, None, f"invalid_refinement: {error}"
    return (
        result,
        warp,
        float(score),
        "ecc_returned_estimate_not_geometric_verification",
    )


def query_database(
    database: LineDatabase,
    mask: Array,
    *,
    top_k: int = 5,
    truncation: float = 12.0,
    iterations: int = 100,
    epsilon: float = 1e-6,
) -> list[Match]:
    if type(top_k) is not int or not 1 <= top_k <= database.config.count:
        raise ValueError("top_k must be within database count")
    query, scale = resize_query(mask, (database.config.width, database.config.height))
    distances = np.linalg.norm(database.descriptors - descriptor(query), axis=1)
    indices = np.argsort(distances, kind="stable")[:top_k]
    unscale = np.linalg.inv(scale)
    result = []
    for index in indices:
        h, warp, score, reason = refine(
            database.masks[index],
            query,
            database.H[index],
            truncation=truncation,
            iterations=iterations,
            epsilon=epsilon,
        )
        result.append(
            Match(
                int(index),
                float(distances[index]),
                normalize_homography(unscale @ database.H[index]),
                h is not None,
                reason,
                score,
                None if h is None else normalize_homography(unscale @ h),
                None if warp is None else normalize_homography(warp @ scale),
            )
        )
    return result
