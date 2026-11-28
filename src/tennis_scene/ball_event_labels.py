"""Ball-event labels shared by SceneResult and the analytics store."""

from typing import Literal

EventSource = Literal["annotation", "rule", "refiner"]
EVENT_SOURCES = ("annotation", "rule", "refiner")
EVENT_STATUS = ("certain", "probable", "uncertain")

EVENT_CLASSES_V4 = ("shot", "bounce", "net", "unknown")
EVENT_SUBTYPES_V4 = (
    "serve_overhead",
    "serve_underhand",
    "groundstroke_forehand",
    "groundstroke_backhand",
    "volley_forehand",
    "volley_backhand",
    "volley_center",
    "smash",
    "service_let",
    "net_in",
    "net_fault",
    "bounce",
    "unknown",
)
EVENT_SUBTYPES_BY_KIND_V4 = (
    (0, 1, 2, 3, 4, 5, 6, 7),  # shot
    (11,),  # bounce
    (8, 9, 10),  # net
    (12,),  # unknown
)
