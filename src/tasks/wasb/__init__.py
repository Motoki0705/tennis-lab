"""WASB-based tennis dataset augmentation utilities.

This package provides tools for semi-automatic tennis ball annotation
and dataset expansion using WASB/HRCNet models.

Main components:
- inference: Ball detection predictors (WASBPredictor, HRCNetWASBPredictor)
- data: Video extraction utilities
- models: Clip segmentation models
- pipeline: End-to-end annotation pipeline
- tennis_format: Label.csv I/O helpers
"""

from .tennis_format import (
    TennisLabelRow,
    load_label_csv,
    make_empty_row,
    row_from_detection,
    save_label_csv,
)

__all__ = [
    "TennisLabelRow",
    "load_label_csv",
    "save_label_csv",
    "make_empty_row",
    "row_from_detection",
]
