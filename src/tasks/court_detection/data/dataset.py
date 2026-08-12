"""The sole production Dataset for Court detection."""

from __future__ import annotations

from torch.utils.data import Dataset

from src.tasks.court_detection.data.contracts import CourtSampleRecord
from src.tasks.court_detection.data.processing.pipeline import CourtProcessingPipeline


class CourtDetectionDataset(Dataset[dict[str, object]]):
    """Delegate source loading and target construction to one resolved pipeline."""

    def __init__(
        self,
        records: tuple[CourtSampleRecord, ...],
        *,
        pipeline: CourtProcessingPipeline,
    ) -> None:
        if not records:
            raise ValueError("CourtDetectionDataset requires a non-empty record set.")
        self.records = records
        self.pipeline = pipeline
        self.pipeline.preflight(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.pipeline.process(self.records[index])


__all__ = ["CourtDetectionDataset"]
