"""Check visible paint endings; static net tape often extends past court ends.

No camera identity, exclusion band, or reference coordinates are accepted.
Support outside the finite template segment is measured, not inferred from a
search window clipped to the segment itself.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from ..homography import COURT_HOMOGRAPHY_EDGES, court_template_xy
from ..hybrid_homography import project_candidate
from ..line_evidence import FloatArray
from ..paint_observations import PaintConfig, stripe_pairs


def measure_endings(image: NDArray[np.uint8], matrix: FloatArray, index: int,
                    config: PaintConfig) -> tuple[tuple[str, str], dict[str, object]]:
    """Five probes inside/outside each end; missing pixels remain unobservable.

    Fractions avoid the intersection itself and the paint cap. At least three
    probes must be observable on each side. A white stripe must be present
    inside before absence outside can certify a finite ground marking.
    """
    projected = project_candidate(matrix, court_template_xy().astype(float))
    if projected is None:
        raise ValueError("Invalid homography for endpoint inspection")
    start, end = projected[np.asarray(COURT_HOMOGRAPHY_EDGES[index])]
    delta = end - start
    normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
    locations = np.concatenate([np.linspace(-.12, -.04, 5), np.linspace(.04, .12, 5),
                                np.linspace(.88, .96, 5), np.linspace(1.04, 1.12, 5)])
    centres = start + locations[:, None] * delta
    offsets = np.arange(-12., 12.01, config.profile_step_px)
    xy = centres[:, None] + offsets[None, :, None] * normal
    height, width = image.shape[:2]
    valid = ((xy[..., 0] >= 1) & (xy[..., 0] < width - 2) & (xy[..., 1] >= 1) & (xy[..., 1] < height - 2)).all(axis=1)
    profiles = cv2.remap(image.min(axis=2).astype(np.float32), xy[..., 0].astype(np.float32),
                         xy[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    present = np.array([bool(valid[i] and any(abs(offset) <= config.max_line_residual_px
                                             for offset, _ in stripe_pairs(profile, offsets, config)))
                        for i, profile in enumerate(profiles)])
    statuses = []
    for outside, inside in ((slice(0, 5), slice(5, 10)), (slice(15, 20), slice(10, 15))):
        if min(int(valid[inside].sum()), int(valid[outside].sum())) < 3:
            statuses.append("unobservable")
            continue
        inner = float(present[inside][valid[inside]].mean())
        outer = float(present[outside][valid[outside]].mean())
        statuses.append("unobservable" if inner < .6 else "ended" if outer <= .2 else "continues" if outer >= .6 else "ambiguous")
    return (statuses[0], statuses[1]), {"statuses": statuses, "sample_t": locations.tolist(),
                                      "observable": valid.tolist(), "stripe_present": present.tolist()}
