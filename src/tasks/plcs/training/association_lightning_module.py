"""PLCS association training in the standard task lifecycle."""

from typing import Any

from src.tasks.base.training.association_module import AssociationLightningBase
from src.tasks.plcs.model_io.factory import compose_plcs_association_model_io


class PLCSAssociationLightningModule(AssociationLightningBase):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model_io = compose_plcs_association_model_io(config)
        self.model = self.model_io.model
