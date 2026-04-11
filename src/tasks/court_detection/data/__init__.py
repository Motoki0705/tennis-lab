"""Data loading for court detection."""

from src.tasks.court_detection.data.court_kp_dataset import CourtKPDataset
from src.tasks.court_detection.data.court_line_dataset import CourtLineDataset
from src.tasks.court_detection.data.court_seg_dataset import CourtSegDataset

__all__ = ["CourtKPDataset", "CourtLineDataset", "CourtSegDataset"]
