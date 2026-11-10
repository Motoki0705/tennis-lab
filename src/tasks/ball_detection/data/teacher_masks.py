"""Explicit adopted teacher eligibility, without changing visual-review facts.

255 marks a legacy clip whose original point-kind/training-policy interpretation is
preserved. It is a declared storage mode, not a fallback for missing mask columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .store import BallFrameStore

EXPLICIT_MASK_SCHEMA: Final = "ball_detection_frames.v2"
MASK_FIELDS: Final = ("teacher_heatmap", "teacher_coordinate")
IGNORE: Final = 0
NEGATIVE: Final = 1
POSITIVE: Final = 2
LEGACY: Final = 255


@dataclass(frozen=True)
class FrameTeacherMasks:
    heatmap: NDArray[np.uint8]
    # Coordinate: 0 = excluded, 1 = single observed ball, 255 = legacy.
    coordinate: NDArray[np.uint8]

    @classmethod
    def from_columns(
        cls, columns: Mapping[str, NDArray[np.generic]], count: int
    ) -> FrameTeacherMasks:
        values = []
        for name in MASK_FIELDS:
            if name not in columns:
                raise ValueError(f"Explicit ball store masks require column {name}")
            value = columns[name]
            if value.dtype != np.uint8 or value.shape != (count,):
                raise ValueError(
                    f"Ball teacher mask {name} must be uint8 with one value per frame"
                )
            values.append(value)
        result = cls(values[0], values[1])
        if (
            not np.isin(result.heatmap, [IGNORE, NEGATIVE, POSITIVE, LEGACY]).all()
            or not np.isin(result.coordinate, [0, 1, LEGACY]).all()
        ):
            raise ValueError("Ball store contains an unknown teacher mask code")
        if not np.array_equal(result.heatmap == LEGACY, result.coordinate == LEGACY):
            raise ValueError("Legacy ball masks must declare both heads together")
        return result

    @property
    def explicit(self) -> NDArray[np.bool_]:
        result: NDArray[np.bool_] = self.heatmap != LEGACY
        return result

    def validate(self, store: BallFrameStore) -> None:
        from .store import POINT_KIND_CODES

        explicit = self.explicit
        for clip in store.clips:
            rows = store.clip_rows(clip)
            if explicit[rows].any() != explicit[rows].all():
                raise ValueError("A ball clip cannot mix legacy and adopted mask modes")
        counts = store.frames["inst_count"]
        owners: NDArray[np.int64] = np.repeat(
            np.arange(len(store), dtype=np.int64), counts
        )
        kinds = store.instances["point_kind"]
        observed: NDArray[np.int64] = np.bincount(
            owners[kinds == POINT_KIND_CODES["observed"]], minlength=len(store)
        )
        absent: NDArray[np.int64] = np.bincount(
            owners[kinds == POINT_KIND_CODES["out_of_frame"]], minlength=len(store)
        )
        eligible = explicit & (self.heatmap != IGNORE)
        if (eligible & (~store.frames["annotated"] | ~store.frames["is_target"])).any():
            raise ValueError("Adopted teachers must be reviewed target frames")
        if (eligible & (observed + absent != counts)).any():
            raise ValueError(
                "Estimated or unresolved ball instances cannot be adopted teachers"
            )
        if (explicit & (self.heatmap == NEGATIVE) & (observed != 0)).any():
            raise ValueError("A negative ball frame cannot contain an observed ball")
        if (explicit & (self.heatmap == POSITIVE) & (observed == 0)).any():
            raise ValueError("A positive ball frame needs an observed ball")
        expected_coordinate = eligible & (observed == 1) & (counts == 1)
        if not np.array_equal(
            self.coordinate[explicit] == 1, expected_coordinate[explicit]
        ):
            raise ValueError(
                "Adopted coordinate mask must select exactly one eligible observed ball"
            )

    def restrict_heatmaps(self, supervised: NDArray[np.bool_]) -> NDArray[np.bool_]:
        result: NDArray[np.bool_] = supervised & (self.heatmap != IGNORE)
        return result
