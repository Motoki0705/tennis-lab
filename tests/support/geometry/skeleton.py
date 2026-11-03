"""Hand specified COCO17 geometry; body lengths are in metres."""

import torch
from torch import Tensor


def coco_pose_m() -> Tensor:
    # Explicit COCO body ordering, independent of the production edge tables:
    # shoulders 5/6, elbows 7/8, wrists 9/10, hips 11/12,
    # knees 13/14, ankles 15/16. Face points do not enter these APIs.
    return torch.tensor(
        [
            [1.5, 0.0, 6.0],
            [1.0, 0.0, 6.5],
            [2.0, 0.0, 6.5],
            [0.5, 0.0, 6.0],
            [2.5, 0.0, 6.0],
            [0.0, 0.0, 4.0],
            [3.0, 0.0, 4.0],
            [0.0, 3.0, 4.0],
            [3.0, -4.0, 4.0],
            [0.0, 6.0, 8.0],
            [3.0, -4.0, -2.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [3.0, 5.0, 0.0],
            [4.0, 3.0, 0.0],
            [3.0, 5.0, 12.0],
        ],
        dtype=torch.float64,
    )
