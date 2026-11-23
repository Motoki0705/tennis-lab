"""Request-boundary tests for repository-owned GVHMR mesh recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from src.submodules.configuration import SmplAssetPaths
from src.submodules.models import GvhmrMeshRecovery, GvhmrRequest, GvhmrResult
from src.submodules.models.gvhmr import mesh_recovery
from src.submodules.vendor.gvhmr import pipeline
from src.utils.models.tensorrt_artifacts import EngineBundle


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

    model._device = torch.device("cpu")
    result = model._predict_impl(request)

    assert isinstance(result, GvhmrResult)
    assert len(vendor.calls) == 1
    assert vendor.calls[0]["kp2d"].shape == (3, 17, 3)
    assert vendor.calls[0]["bbx_xys"].shape == (3, 3)
    assert vendor.calls[0]["f_imgseq"].shape == (3, 1024)


@pytest.mark.parametrize("corrupted", [None, "length", "mask"])
def test_tensorrt_sequence_validation_precedes_fp32_decode(
    corrupted: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        torch.testing.assert_close(inputs["length"], torch.tensor([3, 3]))
        torch.testing.assert_close(inputs["f_imgseq"][:, 0, 0], torch.tensor([0.0, 4.0]))
        camera = torch.tensor([1.5, 3.5], dtype=torch.float16).view(2, 1, 1).expand(2, 3, 3)
        return {
            "length_out": torch.tensor([3, 2 if corrupted == "length" else 3]),
            "mask_out": torch.tensor([[True, True, True], [True, corrupted != "mask", True]]),
            "pred_x": torch.full((2, 3, 151), 0.25, dtype=torch.float16),
            "pred_cam": camera,
            "static_conf_logits": torch.full((2, 3, 6), 2.0, dtype=torch.float16),
        }

    decoder_calls: list[dict[str, torch.Tensor]] = []

    def decode(
        _pipeline: pipeline.Pipeline,
        _inputs: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        *,
        static_cam: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        assert static_cam is True
        decoder_calls.append(prediction)
        assert prediction.keys() == {"pred_x", "pred_cam", "static_conf_logits"}
        assert all(value.dtype == torch.float32 for value in prediction.values())
        parameters = {
            "body_pose": torch.zeros((1, 3, 63)),
            "betas": torch.zeros((1, 3, 10)),
            "global_orient": torch.zeros((1, 3, 3)),
            "transl": prediction["pred_cam"],
        }
        return {
            "pred_smpl_params_incam": parameters,
            "pred_smpl_params_global": parameters,
        }

    monkeypatch.setattr(mesh_recovery, "TensorRTRunner", lambda _bundle, _device: run)
    monkeypatch.setattr(pipeline, "EnDecoder", lambda **_options: torch.nn.Module())
    monkeypatch.setattr(pipeline.Pipeline, "decode_prediction", decode)
    model = GvhmrMeshRecovery.__new__(GvhmrMeshRecovery)
    model.checkpoint = tmp_path / "gvhmr.ckpt"
    model.checkpoint.touch()
    model.body_models_dir = tmp_path
    model.smpl_assets = SmplAssetPaths(
        hmr2_mean_params=tmp_path / "mean.npz",
        smplx_to_smpl=tmp_path / "convert.pt",
        smpl_coco17_regressor=tmp_path / "coco17.pt",
        smplx_verts437=tmp_path / "vertices.pt",
        smpl_neutral_joint_regressor=tmp_path / "joints.pt",
    )
    model.engine = cast(EngineBundle, object())
    model._device = torch.device("cpu")
    model._load_impl()
    monkeypatch.setattr(model, "load", lambda: None)
    first, second = _request(), _request()
    first.f_imgseq.zero_()
    second.f_imgseq.fill_(4.0)

    if corrupted is not None:
        with pytest.raises(ValueError, match="changed the complete sequence length or mask"):
            model.predict_batch((first, second))
        assert decoder_calls == []
    else:
        results = model.predict_batch((first, second))
        assert len(decoder_calls) == 2
        for result, value in zip(results, (1.5, 3.5), strict=True):
            torch.testing.assert_close(result.smpl_params_incam["transl"], torch.full((3, 3), value))
