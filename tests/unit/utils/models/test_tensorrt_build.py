"""Profile registration with TensorRT's void setter and dimension readback."""

from __future__ import annotations

import pytest

from src.utils.models.tensorrt_artifacts import InputProfile, Shape
from src.utils.models.tensorrt_build import set_profile_shapes


@pytest.mark.parametrize("discard_update", [False, True])
def test_profile_registration_checks_the_dimensions(discard_update: bool) -> None:
    profile = InputProfile(minimum=(1, 2, 6), optimum=(1, 145, 6), maximum=(1, 177, 6))

    class BuilderProfile:
        def __init__(self) -> None:
            self.tensors: dict[str, tuple[Shape, ...]] = {}

        def set_shape(
            self, name: str, minimum: Shape, optimum: Shape, maximum: Shape
        ) -> None:
            if not discard_update:
                self.tensors[name] = (minimum, optimum, maximum)

        def get_shape(self, name: str) -> list[Shape]:
            return list(self.tensors.get(name, ()))

    builder_profile = BuilderProfile()
    if discard_update:
        with pytest.raises(ValueError, match="TensorRT profile f_cam_angvel"):
            set_profile_shapes(builder_profile, {"f_cam_angvel": profile})
    else:
        set_profile_shapes(builder_profile, {"f_cam_angvel": profile})
        assert builder_profile.get_shape("f_cam_angvel") == [
            (1, 2, 6),
            (1, 145, 6),
            (1, 177, 6),
        ]
