"""BLCS view-side and clip identity model; global MHA with mHC."""

from src.tasks.base.models.view_association import (
    ViewQueryAssociationModel,
    ViewQueryModelConfig,
)


class BLCSViewAssociationModel(ViewQueryAssociationModel):
    def __init__(self, config: ViewQueryModelConfig) -> None:
        super().__init__(config, num_keypoints=1)
