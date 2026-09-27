"""Source adapters: each parses its own annotation format into ``ClipSpec`` values."""

from src.tasks.ball_detection.generate_dataset.frame_store.sources.chat_annotation import (
    ChatAnnotationSourceConfig,
    collect_chat_annotation,
)
from src.tasks.ball_detection.generate_dataset.frame_store.sources.meiji import (
    MeijiSourceConfig,
    collect_meiji,
)
from src.tasks.ball_detection.generate_dataset.frame_store.sources.tracknet import (
    TrackNetSourceConfig,
    collect_tracknet,
)

__all__ = [
    "ChatAnnotationSourceConfig",
    "MeijiSourceConfig",
    "TrackNetSourceConfig",
    "collect_chat_annotation",
    "collect_meiji",
    "collect_tracknet",
]
