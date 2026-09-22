"""PLCS association loader composition."""

from src.tasks.base.data.association_datamodule import AssociationDataModuleBase
from src.tasks.plcs.data.association_dataset import PLCSAssociationDataset


class PLCSAssociationDataModule(AssociationDataModuleBase):
    dataset_type = PLCSAssociationDataset

    def _dataset_name(self) -> str:
        return "plcs"
