"""Render selected dense targets at the augmented training resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import cv2
import numpy as np
import torch
from torch import Tensor

from src.tasks.court_detection.data.contracts import (
    CourtDenseTargetKind,
    CourtRawSample,
)
from src.tasks.court_detection.data.target_generation.line import generate_line_target
from src.tasks.court_detection.data.target_generation.rasterization import (
    CourtPlaneRasterizer,
)
from src.tasks.court_detection.data.target_generation.segmentation import (
    generate_segmentation_target,
)
from src.tasks.court_detection.data.target_generation.semantic_line import (
    generate_semantic_line_target,
)
from src.tasks.court_detection.target_schemas import (
    LINE_TARGET_SCHEMA,
    SEGMENTATION_TARGET_SCHEMA,
    SEMANTIC_LINE_TARGET_SCHEMA,
    line_target_definition,
)


def generate_online_targets(
    raw: CourtRawSample,
    schemas: Mapping[CourtDenseTargetKind, str],
    *,
    source_to_output: Tensor | None = None,
    output_size_hw: tuple[int, int] | None = None,
    content_size_hw: tuple[int, int] | None = None,
) -> dict[CourtDenseTargetKind, Tensor]:
    """Fit H in source pixels once per court and share A@H across all heads.

    Class flips belong to the target builders. This renderer uses the original
    camera-view channel order, including when A contains a horizontal flip.
    """
    if not schemas:
        return {}
    if len(raw.court_instances) != 1 and any(
        kind in {"seg", "semantic_line"} or schema == LINE_TARGET_SCHEMA
        for kind, schema in schemas.items()
    ):
        raise ValueError(
            "Current single-court dense targets require exactly one selected court."
        )
    height, width = output_size_hw or (raw.image.height, raw.image.width)
    matrix = (
        np.eye(3)
        if source_to_output is None
        else source_to_output.detach().cpu().numpy().astype(np.float64)
    )
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Online target geometry must be a finite 3x3 matrix.")
    rasterizers: list[CourtPlaneRasterizer | None] = []
    for instance in raw.court_instances:
        original = CourtPlaneRasterizer.from_instance(
            instance, width=raw.image.width, height=raw.image.height
        )
        if original is None:
            rasterizers.append(None)
            continue
        points_h = (
            np.concatenate((original.image_points, np.ones((14, 1))), axis=1) @ matrix.T
        )
        if np.any(np.abs(points_h[:, 2]) < 1e-10):
            raise ValueError("Online target transformation crosses a keypoint horizon.")
        rasterizers.append(
            replace(
                original,
                width=width,
                height=height,
                homography=matrix @ original.homography,
                image_points=points_h[:, :2] / points_h[:, 2, None],
            )
        )
    projectors = tuple(rasterizers)
    support = cv2.warpPerspective(
        np.ones((raw.image.height, raw.image.width), dtype=np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    if content_size_hw is not None:
        support[content_size_hw[0] :, :] = 0
        support[:, content_size_hw[1] :] = 0
    result: dict[CourtDenseTargetKind, Tensor] = {}
    for kind, schema in schemas.items():
        if kind == "seg":
            if schema != SEGMENTATION_TARGET_SCHEMA:
                raise ValueError(f"Unsupported online SEG schema: {schema}.")
            array = generate_segmentation_target(
                height=height,
                width=width,
                instances=raw.court_instances,
                rasterizers=projectors,
            )
        elif kind == "line":
            definition = line_target_definition(schema)
            array = generate_line_target(
                height=height,
                width=width,
                instances=raw.court_instances,
                rasterizers=projectors,
                line_width_metres=definition.line_width_metres,
                baseline_width_metres=definition.baseline_width_metres,
            )
        elif kind == "semantic_line":
            if schema != SEMANTIC_LINE_TARGET_SCHEMA:
                raise ValueError(f"Unsupported online semantic LINE schema: {schema}.")
            channels = raw.keypoint_channels
            if channels is None or channels.physical_indices.shape != (14, 1):
                raise ValueError("Semantic LINE requires ordered singleton KP14.")
            array = generate_semantic_line_target(
                height=height,
                width=width,
                instances=raw.court_instances,
                semantic_to_physical=channels.physical_indices[:, 0],
                rasterizers=projectors,
            )
        else:
            raise ValueError(f"Unknown dense target kind: {kind}.")
        array[support == 0] = 0
        tensor = torch.from_numpy(array)
        result[kind] = (
            (tensor > 0).float().unsqueeze(0) if kind == "line" else tensor.long()
        )
    return result
