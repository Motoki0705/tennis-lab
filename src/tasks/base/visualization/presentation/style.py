"""Color and notation definitions shared by presentation renderers."""

from colorsys import hsv_to_rgb, rgb_to_hsv
from typing import Final

from matplotlib.colors import to_rgb

BACKGROUND: Final = "#121212"
FOREGROUND: Final = "#F1F3F5"
TRAINING: Final = "#E76F51"
FROZEN: Final = "#457B9D"
NEW: Final = "#8AC926"
FIXED: Final = "#6C757D"

# Court learning curves and the KP + LINE inference handoff share this palette.
COURT_HEAD_COLORS: Final = {
    "kp": "#35B779",  # viridis green
    "seg": "#B48EF5",
    "line": "#4CC9F0",
    "semline": "#FFD166",
    "pose": "#EF8FB2",
}
SYNTHETIC: Final = "#E76F51"
REAL: Final = "#4CC9F0"
CORRECTION_UNREVIEWED: Final = "#495057"
CORRECTION_RERANK: Final = "#FFD166"
CORRECTION_MOVED: Final = "#F4A261"
CORRECTION_ADDED: Final = NEW
CORRECTION_REMOVED: Final = "#E63946"

REMOVED_ALPHA: Final = 0.2
# The state API has no per-block base color. The supported removal use case is
# a training block; a canonical color also keeps standalone endpoints stable.
_training_hsv = rgb_to_hsv(*to_rgb(TRAINING))
REMOVED_RGB: Final = hsv_to_rgb(
    _training_hsv[0], _training_hsv[1] * 0.2, _training_hsv[2]
)
CONCEPT_BADGE: Final = "構想"
CONCEPT_LINESTYLE: Final = "--"

# Include Japanese glyphs without changing matplotlib's global rcParams.
FONT_FAMILY: Final = ("DejaVu Sans", "Noto Sans CJK JP")
