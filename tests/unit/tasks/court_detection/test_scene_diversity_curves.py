"""The final fit history must not inherit the lost scene2 attempt tail."""

from pathlib import Path
from runpy import run_path

import pytest


def test_resumed_history_excludes_discarded_tail_and_prefers_new_boundary():
    script = Path("knowledge/runs/run-court-scene-diversity-colab-s42/export_curves.py")
    select = run_path(str(script))["select_points"]
    retained, discarded = select(
        [[26500, 1.0], [26528, 2.0], [27150, 0.1]],
        [[26528, 2.1], [26600, 1.2], [33160, 0.5]],
    )
    assert retained == [[26500, 1.0], [26528, 2.1], [26600, 1.2], [33160, 0.5]]
    assert discarded == [[27150, 0.1]]
    with pytest.raises(ValueError, match="precedes"):
        select([[26500, 1.0]], [[26527, 1.0]])
    with pytest.raises(ValueError, match="Nonfinite"):
        select([[33160, float("nan")]])
