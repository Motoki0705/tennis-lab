"""Tests for src/submodules/models/_base/inference_model.py."""

import pytest
import torch

from src.submodules.models import BaseInferenceModel
from src.utils.device import DeviceSelectionError


class DoublingModel(BaseInferenceModel[int, int]):
    """Minimal concrete model for lifecycle testing."""

    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.load_calls = 0
        self.unload_calls = 0
        self.grad_enabled_during_predict: bool | None = None

    def _load_impl(self) -> None:
        self.load_calls += 1

    def _unload_impl(self) -> None:
        self.unload_calls += 1

    def _predict_impl(self, request: int) -> int:
        self.grad_enabled_during_predict = torch.is_grad_enabled()
        return request * 2


class TestBaseInferenceModel:
    def test_load_is_idempotent(self):
        model = DoublingModel()
        assert not model.is_loaded
        model.load()
        model.load()
        assert model.is_loaded
        assert model.load_calls == 1

    def test_predict_autoloads_and_disables_grad(self):
        model = DoublingModel()
        assert model.predict(21) == 42
        assert model.is_loaded
        assert model.load_calls == 1
        assert model.grad_enabled_during_predict is False

    def test_predict_is_the_only_inference_entrypoint(self):
        assert not callable(DoublingModel())

    def test_unload_resets_state(self):
        model = DoublingModel()
        model.load()
        model.unload()
        model.unload()  # idempotent
        assert not model.is_loaded
        assert model.unload_calls == 1
        # predict loads again
        assert model.predict(1) == 2
        assert model.load_calls == 2

    def test_device_resolution(self):
        model = DoublingModel()
        assert model.device == torch.device("cpu")

    def test_explicit_cuda_does_not_fall_back_to_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(DeviceSelectionError, match="CUDA is unavailable"):
            BaseInferenceModel.__init__(DoublingModel.__new__(DoublingModel), "cuda")
