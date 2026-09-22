"""PLCS view-side and clip identity model; global MHA with mHC."""

from src.utils.models.components.view_query import (
    ViewQueryAssociationModel,
    ViewQueryModelConfig,
)


class PLCSViewAssociationModel(ViewQueryAssociationModel):
    def __init__(self, config: ViewQueryModelConfig) -> None:
        super().__init__(config, num_keypoints=17)
