"""Request-boundary tests for repository-owned GVHMR mesh recovery."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from src.submodules.models import GvhmrMeshRecovery, GvhmrRequest, GvhmrResult


def _request(num_frames: int = 3) -> GvhmrRequest:
    keypoints = torch.zeros((num_frames, 17, 3), dtype=torch.float32)
    keypoints[..., 2] = 1.0
    boxes = torch.ones((num_frames, 3), dtype=torch.float32)
    return GvhmrRequest(
        kp2d=keypoints,
        bbx_xys=boxes,
        f_imgseq=torch.zeros((num_frames, 1024), dtype=torch.float32),
        width=1920,
        height=1080,
        static_cam=True,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kp2d", torch.ones(3), "kp2d must have shape"),
        ("bbx_xys", torch.ones((3, 4)), "bbx_xys must have shape"),
        (
            "f_imgseq",
            torch.ones((3, 1024), dtype=torch.float64),
            "f_imgseq must have dtype",
        ),
        (
            "f_imgseq",
            torch.empty((3, 1024), dtype=torch.float32, device="meta"),
            "tensors must share one device",
        ),
    ],
)
def test_request_rejects_rank_shape_and_dtype_before_model_entry(
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    values: dict[str, Any] = {
        "kp2d": torch.zeros((3, 17, 3), dtype=torch.float32),
        "bbx_xys": torch.ones((3, 3), dtype=torch.float32),
        "f_imgseq": torch.zeros((3, 1024), dtype=torch.float32),
        "width": 1920,
        "height": 1080,
        "static_cam": True,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        GvhmrRequest(**values)


def test_request_rejects_nonpositive_box_size() -> None:
    boxes = torch.ones((3, 3), dtype=torch.float32)
    boxes[1, 2] = 0.0

    with pytest.raises(ValueError, match="sizes must be positive"):
        GvhmrRequest(
            kp2d=torch.zeros((3, 17, 3), dtype=torch.float32),
            bbx_xys=boxes,
            f_imgseq=torch.zeros((3, 1024), dtype=torch.float32),
            width=1920,
            height=1080,
            static_cam=True,
        )


def test_predict_revalidates_mutated_request_before_vendor_entry() -> None:
    request = _request()
    object.__setattr__(request, "kp2d", torch.ones(3, dtype=torch.float32))

    class VendorSpy:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, data: object, *, static_cam: bool) -> object:
            del data, static_cam
            self.calls += 1
            raise AssertionError("invalid request reached the vendored model")

    vendor = VendorSpy()
    model = GvhmrMeshRecovery.__new__(GvhmrMeshRecovery)
    model._model = cast(Any, vendor)

    with pytest.raises(ValueError, match="kp2d must have shape"):
        model._predict_impl(request)

    assert vendor.calls == 0


def test_valid_request_reaches_vendor_with_documented_shapes() -> None:
    request = _request()
    params = {
        "body_pose": torch.zeros((3, 63), dtype=torch.float32),
        "betas": torch.zeros((3, 10), dtype=torch.float32),
        "global_orient": torch.zeros((3, 3), dtype=torch.float32),
        "transl": torch.zeros((3, 3), dtype=torch.float32),
    }

    class VendorSpy:
        def __init__(self) -> None:
            self.calls: list[dict[str, torch.Tensor]] = []

        def predict(
            self,
            data: dict[str, torch.Tensor],
            *,
            static_cam: bool,
        ) -> dict[str, object]:
            assert static_cam is True
            self.calls.append(data)
            return {
                "smpl_params_incam": params,
                "smpl_params_global": params,
            }

    vendor = VendorSpy()
    model = GvhmrMeshRecovery.__new__(GvhmrMeshRecovery)
    model._model = cast(Any, vendor)

    result = model._predict_impl(request)

    assert isinstance(result, GvhmrResult)
    assert len(vendor.calls) == 1
    assert vendor.calls[0]["kp2d"].shape == (3, 17, 3)
    assert vendor.calls[0]["bbx_xys"].shape == (3, 3)
    assert vendor.calls[0]["f_imgseq"].shape == (3, 1024)
