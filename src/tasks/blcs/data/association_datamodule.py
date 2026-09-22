"""BLCS association loader composition."""

from src.tasks.base.data.association_datamodule import AssociationDataModuleBase
from src.tasks.blcs.data.association_dataset import BLCSAssociationDataset


class BLCSAssociationDataModule(AssociationDataModuleBase):
    dataset_type = BLCSAssociationDataset

    def _dataset_name(self) -> str:
        return "blcs"
