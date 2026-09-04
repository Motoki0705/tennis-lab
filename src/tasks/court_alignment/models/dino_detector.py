"""LoRA-adapted official DINO detector for oriented tennis courts.

The released COCO checkpoint is loaded strictly into its original 91-class
architecture before any task-specific mutation.  Afterwards the classifier is
replaced by a one-class court classifier, a scale/axial-orientation head is
attached to every decoder layer, and selected linear layers receive LoRA.
Third-party DINO source files are never copied or modified.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch
from torch import Tensor, nn

from src.tasks.court_alignment.models.dino_input import (
    DINO_DEFAULT_MAX_LONG_SIDE,
    DINO_DEFAULT_SHORT_SIDE,
    DinoHeatmapInputAdapter,
    DinoInputMode,
    validate_dino_heatmaps,
)
from src.utils.models.lora import apply_lora, iter_lora_parameters

DINO_COCO_CLASS_COUNT = 91
COURT_CLASS_COUNT = 1
COURT_PARAMETER_COUNT = 3
DEFAULT_DINO_LORA_TARGETS = (
    "qkv",
    "proj",
    "value_proj",
    "output_proj",
    "linear1",
    "linear2",
)

_DINO_MODELS_PACKAGE = "_tennis_lab_court_alignment_dino_models"
NestedTensorFactory = Callable[[Tensor, Tensor], object]


class CourtScaleAxisHead(nn.Module):
    """Predict raw long-side scale and an unoriented long-edge axis.

    The returned final dimension is ``(long_side_logit, axial_x, axial_y)``.
    The shared loss/decoder boundary applies sigmoid to the long-side logit
    and L2-normalises the final two values into
    ``(cos(2 theta), sin(2 theta))``.  Keeping logits raw gives the loss stable
    gradients while the axial representation remains modulo 180 degrees.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, COURT_PARAMETER_COUNT),
        )
        output = self.layers[-1]
        if not isinstance(output, nn.Linear):  # pragma: no cover - construction invariant
            raise RuntimeError("CourtScaleAxisHead output must be linear.")
        with torch.no_grad():
            output.weight.zero_()
            output.bias.copy_(torch.tensor((0.0, 1.0, 0.0)))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return cast(Tensor, self.layers(hidden_states))


def _stack_decoder_hidden_states(output: Any, *, hidden_dim: int) -> Tensor:
    """Stack the official decoder's per-layer ``(B,Q,D)`` state list."""

    if not isinstance(output, tuple) or not output:
        raise RuntimeError("Official DINO transformer returned an invalid output.")
    decoder_states = output[0]
    if not isinstance(decoder_states, list) or not decoder_states:
        raise RuntimeError(
            "Official DINO decoder hidden states must be a non-empty list."
        )
    first_shape: tuple[int, ...] | None = None
    for state in decoder_states:
        if not isinstance(state, Tensor) or state.ndim != 3:
            raise RuntimeError(
                "Every official DINO decoder hidden state must have shape (B,Q,D)."
            )
        if state.shape[-1] != hidden_dim:
            raise RuntimeError(
                "Official DINO decoder hidden size does not match the court head."
            )
        shape = tuple(state.shape)
        if first_shape is None:
            first_shape = shape
        elif shape != first_shape:
            raise RuntimeError(
                "Official DINO decoder layers must share batch/query/hidden shapes."
            )
    return torch.stack(decoder_states, dim=0)


def validate_dino_court_output(output: object) -> Mapping[str, object]:
    """Validate the task-head result after detector ``forward`` completes."""

    if not isinstance(output, Mapping):
        raise TypeError("Official DINO model must return a mapping.")
    logits = output.get("pred_logits")
    boxes = output.get("pred_boxes")
    court_boxes = output.get("pred_court_boxes")
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise RuntimeError("Official DINO logits must have shape (B,Q,C).")
    if not isinstance(boxes, Tensor) or boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise RuntimeError("Official DINO boxes must have shape (B,Q,4).")
    if (
        not isinstance(court_boxes, Tensor)
        or court_boxes.ndim != 3
        or court_boxes.shape[-1] != COURT_PARAMETER_COUNT
    ):
        raise RuntimeError("DINO court parameters must have shape (B,Q,3).")
    if logits.shape[:2] != boxes.shape[:2] or logits.shape[:2] != court_boxes.shape[:2]:
        raise RuntimeError("DINO task heads disagree on batch/query shape.")
    if logits.shape[-1] != COURT_CLASS_COUNT:
        raise RuntimeError(
            f"Court classifier must emit {COURT_CLASS_COUNT} logit, got "
            f"{logits.shape[-1]}."
        )
    auxiliary = output.get("aux_outputs")
    if auxiliary is not None:
        if not isinstance(auxiliary, list):
            raise TypeError("Official DINO aux_outputs must be a list.")
        for layer_output in auxiliary:
            if not isinstance(layer_output, Mapping):
                raise TypeError("Every DINO auxiliary output must be a mapping.")
            layer_logits = layer_output.get("pred_logits")
            layer_boxes = layer_output.get("pred_boxes")
            layer_court_boxes = layer_output.get("pred_court_boxes")
            if (
                not isinstance(layer_logits, Tensor)
                or not isinstance(layer_boxes, Tensor)
                or not isinstance(layer_court_boxes, Tensor)
                or layer_logits.shape != logits.shape
                or layer_boxes.shape != boxes.shape
                or layer_court_boxes.shape != court_boxes.shape
            ):
                raise RuntimeError(
                    "Every DINO auxiliary layer must match the final task-head shapes."
                )
    return output


class DinoCourtDetector(nn.Module):
    """Adapt an already checkpoint-loaded official DINO model for courts.

    Callers that need the released checkpoint should normally use
    :func:`load_pretrained_dino_court_detector`; direct construction exists for
    dependency-injected tests and specialised loaders.  The supplied model
    must still have its original 91-class COCO heads.
    """

    def __init__(
        self,
        dino: nn.Module,
        *,
        input_mode: DinoInputMode,
        short_side: int = DINO_DEFAULT_SHORT_SIDE,
        max_long_side: int = DINO_DEFAULT_MAX_LONG_SIDE,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = DEFAULT_DINO_LORA_TARGETS,
        nested_tensor_factory: NestedTensorFactory | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(dino, nn.Module):
            raise TypeError("dino must be a torch.nn.Module.")
        if getattr(dino, "num_classes", None) != DINO_COCO_CLASS_COUNT:
            raise ValueError(
                "DinoCourtDetector requires the original 91-class COCO model "
                "before court-head replacement."
            )
        if type(lora_rank) is not int or lora_rank <= 0:
            raise ValueError("lora_rank must be a positive integer.")
        if type(lora_alpha) is not float or lora_alpha <= 0.0:
            raise ValueError("lora_alpha must be a positive float.")
        if type(lora_dropout) is not float or not 0.0 <= lora_dropout < 1.0:
            raise ValueError("lora_dropout must be a float in [0, 1).")

        hidden_dim = _dino_hidden_dim(dino)
        self.hidden_dim = hidden_dim
        self.input_adapter = DinoHeatmapInputAdapter(
            mode=input_mode,
            short_side=short_side,
            max_long_side=max_long_side,
        )
        if nested_tensor_factory is not None and not callable(nested_tensor_factory):
            raise TypeError("nested_tensor_factory must be callable or None.")
        self._nested_tensor_factory = nested_tensor_factory
        self.dino = dino
        new_class_modules = _replace_coco_class_heads(dino, hidden_dim=hidden_dim)
        detection_bbox_modules = _dino_detection_bbox_modules(dino)
        self.court_head = CourtScaleAxisHead(hidden_dim)

        # Freeze every pretrained and newly installed DINO parameter first.
        # LoRA factors are created trainable by apply_lora; task detection
        # heads are explicitly re-enabled below.  The backbone and transformer
        # base parameters remain frozen.
        self.dino.requires_grad_(False)
        self.lora_module_names = tuple(
            apply_lora(
                self.dino,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
        )
        gradient_anchor = next(iter_lora_parameters(self.dino), None)
        if gradient_anchor is None:  # pragma: no cover - apply_lora rejects this first
            raise RuntimeError("DINO court model has no LoRA gradient anchor.")
        # A plain tuple keeps an alias to the registered DINO parameter without
        # registering a duplicate state-dict name on this wrapper.
        self._lora_gradient_anchors = (gradient_anchor,)
        for module in (*new_class_modules, *detection_bbox_modules):
            module.requires_grad_(True)

    def forward(
        self,
        heatmaps: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Any]:
        images = self.input_adapter(heatmaps)
        if self.training and not images.requires_grad:
            # Official Swin-L enables re-entrant gradient checkpointing.  Its
            # checkpointed blocks drop parameter gradients when every input is
            # frozen; repeat_rgb/red_only have no trainable input projection.
            # This out-of-place zero anchor activates checkpoint autograd
            # without unfreezing the heatmap, backbone, or transformer base.
            anchor = self._lora_gradient_anchors[0]
            images = images + anchor.reshape(-1)[0] * 0.0
        dino_input: object = images
        if self._nested_tensor_factory is not None:
            # All tensors in this procedural batch share a shape, so the
            # equivalent official NestedTensor mask has no padded pixels.
            # Constructing it directly preserves both the learnable 1x1
            # adapter gradient and the checkpoint anchor above.
            mask = torch.zeros(
                (images.shape[0], images.shape[-2], images.shape[-1]),
                dtype=torch.bool,
                device=images.device,
            )
            dino_input = self._nested_tensor_factory(images, mask)
        captured: dict[str, Tensor] = {}

        def capture_hidden_states(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            captured["decoder"] = torch.stack(output[0], dim=0)

        transformer = cast(Any, self.dino).transformer
        handle = transformer.register_forward_hook(capture_hidden_states)
        try:
            raw_output = self.dino(dino_input, targets)
        finally:
            handle.remove()
        output = dict(raw_output)
        logits = cast(Tensor, output["pred_logits"])
        hidden_states = captured["decoder"]
        query_count = logits.shape[1]

        # DINO prepends denoising queries while training.  Its standard output
        # removes that prefix, so take the same trailing matching-query slice.
        hidden_states = hidden_states[:, :, -query_count:, :]
        court_parameters = self.court_head(hidden_states)
        output["pred_court_boxes"] = court_parameters[-1]

        auxiliary = output.get("aux_outputs")
        if auxiliary is not None:
            augmented_auxiliary: list[dict[str, Any]] = []
            for layer_output, layer_court_parameters in zip(
                cast(list[Mapping[str, object]], auxiliary),
                court_parameters[:-1],
                strict=True,
            ):
                augmented = dict(layer_output)
                augmented["pred_court_boxes"] = layer_court_parameters
                augmented_auxiliary.append(augmented)
            output["aux_outputs"] = augmented_auxiliary
        return output

    def validate_input(self, heatmaps: Tensor) -> None:
        """Validate raw detector evidence before invoking ``forward``."""

        validate_dino_heatmaps(heatmaps)

    def validate_output(self, output: object) -> Mapping[str, object]:
        """Validate task-head tensors after ``forward`` has completed."""

        return validate_dino_court_output(output)

    def trainable_parameter_names(self) -> tuple[str, ...]:
        """Return stable diagnostics for optimizer/config validation."""
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)


def _dino_hidden_dim(dino: nn.Module) -> int:
    hidden_dim = getattr(dino, "hidden_dim", None)
    if type(hidden_dim) is not int:
        transformer = getattr(dino, "transformer", None)
        hidden_dim = getattr(transformer, "d_model", None)
    if type(hidden_dim) is not int or hidden_dim <= 0:
        raise ValueError("Could not determine DINO's positive integer hidden dimension.")
    return hidden_dim


def _new_class_projection(hidden_dim: int) -> nn.Linear:
    projection = nn.Linear(hidden_dim, COURT_CLASS_COUNT)
    prior_probability = 0.01
    prior_bias = -math.log((1.0 - prior_probability) / prior_probability)
    with torch.no_grad():
        projection.bias.fill_(prior_bias)
    return projection


def _replace_coco_class_heads(
    dino: nn.Module,
    *,
    hidden_dim: int,
) -> tuple[nn.Module, ...]:
    class_embed = getattr(dino, "class_embed", None)
    if not isinstance(class_embed, nn.ModuleList) or not class_embed:
        raise ValueError("Official DINO model must expose a non-empty class_embed ModuleList.")
    if any(
        not isinstance(layer, nn.Linear)
        or layer.in_features != hidden_dim
        or layer.out_features != DINO_COCO_CLASS_COUNT
        for layer in class_embed
    ):
        raise ValueError("DINO class_embed does not match the 91-class COCO architecture.")

    replacements: dict[int, nn.Linear] = {}

    def replace(old: nn.Linear) -> nn.Linear:
        existing = replacements.get(id(old))
        if existing is not None:
            return existing
        new = _new_class_projection(hidden_dim)
        new.to(device=old.weight.device, dtype=old.weight.dtype)
        replacements[id(old)] = new
        return new

    decoder_heads = nn.ModuleList(
        [replace(cast(nn.Linear, layer)) for layer in class_embed]
    )
    cast(Any, dino).class_embed = decoder_heads
    transformer = getattr(dino, "transformer", None)
    if not isinstance(transformer, nn.Module):
        raise ValueError("Official DINO model must expose its transformer module.")
    decoder = getattr(transformer, "decoder", None)
    if not isinstance(decoder, nn.Module):
        raise ValueError("Official DINO transformer must expose its decoder module.")
    cast(Any, decoder).class_embed = decoder_heads

    new_modules: list[nn.Module] = list(dict.fromkeys(decoder_heads))
    encoder_head = getattr(transformer, "enc_out_class_embed", None)
    if encoder_head is not None:
        if (
            not isinstance(encoder_head, nn.Linear)
            or encoder_head.in_features != hidden_dim
            or encoder_head.out_features != DINO_COCO_CLASS_COUNT
        ):
            raise ValueError("DINO encoder class head does not match COCO checkpoint.")
        new_encoder_head = replace(encoder_head)
        cast(Any, transformer).enc_out_class_embed = new_encoder_head
        if new_encoder_head not in new_modules:
            new_modules.append(new_encoder_head)

    label_encoder = getattr(dino, "label_enc", None)
    if not isinstance(label_encoder, nn.Embedding) or label_encoder.embedding_dim != hidden_dim:
        raise ValueError("Official DINO model must expose its label_enc embedding.")
    new_label_encoder = nn.Embedding(COURT_CLASS_COUNT + 1, hidden_dim)
    new_label_encoder.to(device=label_encoder.weight.device, dtype=label_encoder.weight.dtype)
    cast(Any, dino).label_enc = new_label_encoder
    cast(Any, dino).num_classes = COURT_CLASS_COUNT
    if hasattr(dino, "dn_labelbook_size"):
        cast(Any, dino).dn_labelbook_size = COURT_CLASS_COUNT
    new_modules.append(new_label_encoder)
    return tuple(new_modules)


def _dino_detection_bbox_modules(dino: nn.Module) -> tuple[nn.Module, ...]:
    """Collect decoder/encoder bbox heads, deduplicating official aliases."""
    bbox_embed = getattr(dino, "bbox_embed", None)
    if not isinstance(bbox_embed, nn.ModuleList) or not bbox_embed:
        raise ValueError("Official DINO model must expose a non-empty bbox_embed ModuleList.")
    if any(not isinstance(layer, nn.Module) for layer in bbox_embed):
        raise ValueError("Every DINO decoder bbox head must be a torch module.")

    transformer = getattr(dino, "transformer", None)
    decoder = getattr(transformer, "decoder", None)
    decoder_bbox_embed = getattr(decoder, "bbox_embed", None)
    if not isinstance(decoder_bbox_embed, nn.ModuleList) or not decoder_bbox_embed:
        raise ValueError("Official DINO decoder must expose bbox_embed aliases.")
    encoder_bbox_embed = getattr(transformer, "enc_out_bbox_embed", None)
    if not isinstance(encoder_bbox_embed, nn.Module):
        raise ValueError("Official two-stage DINO must expose enc_out_bbox_embed.")

    unique: list[nn.Module] = []
    seen: set[int] = set()
    for module in (bbox_embed, decoder_bbox_embed, encoder_bbox_embed):
        if id(module) not in seen:
            unique.append(module)
            seen.add(id(module))
    return tuple(unique)


def _validate_absolute_file(path: Path, *, name: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib.Path.")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{name} must be an ordinary file: {path}")
    return path


def _load_package(
    package_name: str,
    init_path: Path,
) -> ModuleType:
    existing = sys.modules.get(package_name)
    if existing is not None:
        loaded_file = getattr(existing, "__file__", None)
        if loaded_file is None or Path(loaded_file).resolve() != init_path.resolve():
            raise RuntimeError(
                f"Package {package_name!r} is already loaded from {loaded_file!r}."
            )
        return existing
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create an import spec for {init_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        raise
    return module


def _load_official_dino_components(
    repository: Path,
) -> tuple[Callable[[Any], tuple[Any, ...]], type[Any], NestedTensorFactory]:
    if not isinstance(repository, Path) or not repository.is_absolute():
        raise ValueError("DINO repository must be an absolute pathlib.Path.")
    models_init = repository / "models/__init__.py"
    util_init = repository / "util/__init__.py"
    config_path = repository / "config/DINO/DINO_4scale_swin.py"
    missing = [path for path in (models_init, util_init, config_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "DINO git submodule is not initialized or incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )
    _load_package("util", util_init)
    try:
        models = _load_package(_DINO_MODELS_PACKAGE, models_init)
    except ModuleNotFoundError as error:
        if error.name != "MultiScaleDeformableAttention":
            raise
        project_root = repository.parent.parent
        raise RuntimeError(
            "DINO CUDA extension 'MultiScaleDeformableAttention' is missing. "
            f"Build the compatibility-patched extension with: cd {project_root} && "
            "TENNIS_LAB_BUILD_CUDA_OPS=1 .venv/bin/python setup.py "
            "build_ext --inplace. "
            f"Then expose it with: export PYTHONPATH={project_root}:$PYTHONPATH"
        ) from error
    build_dino = getattr(models, "build_dino", None)
    if not callable(build_dino):
        raise RuntimeError(f"Official DINO module has no build_dino: {models_init}")
    slconfig_module = importlib.import_module("util.slconfig")
    slconfig = getattr(slconfig_module, "SLConfig", None)
    if not isinstance(slconfig, type):
        raise RuntimeError("Official DINO util.slconfig has no SLConfig type.")
    misc_module = importlib.import_module("util.misc")
    nested_tensor = getattr(misc_module, "NestedTensor", None)
    if not isinstance(nested_tensor, type):
        raise RuntimeError("Official DINO util.misc has no NestedTensor type.")
    return (
        cast(Callable[[Any], tuple[Any, ...]], build_dino),
        slconfig,
        cast(NestedTensorFactory, nested_tensor),
    )


def _validate_checkpoint_payload(payload: Any, *, checkpoint_path: Path) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"DINO checkpoint root must be a mapping: {checkpoint_path}")
    model_state = payload.get("model")
    checkpoint_args = payload.get("args")
    if not isinstance(model_state, Mapping) or checkpoint_args is None:
        raise ValueError("DINO checkpoint must contain 'model' and 'args'.")
    backbone = getattr(checkpoint_args, "backbone", None)
    if backbone != "swin_L_384_22k":
        raise ValueError(
            "Unsupported DINO checkpoint backbone; expected 'swin_L_384_22k', "
            f"got {backbone!r}."
        )
    return cast(Mapping[str, Any], model_state)


def load_pretrained_dino_court_detector(
    *,
    repository: Path,
    checkpoint_path: Path,
    device: str | torch.device,
    input_mode: DinoInputMode,
    short_side: int = DINO_DEFAULT_SHORT_SIDE,
    max_long_side: int = DINO_DEFAULT_MAX_LONG_SIDE,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_target_modules: Sequence[str] = DEFAULT_DINO_LORA_TARGETS,
) -> DinoCourtDetector:
    """Strict-load released Swin-L DINO, then install court heads and LoRA."""
    checkpoint_path = _validate_absolute_file(checkpoint_path, name="DINO checkpoint")
    build_dino, slconfig, nested_tensor_factory = _load_official_dino_components(
        repository
    )
    config_path = repository / "config/DINO/DINO_4scale_swin.py"
    args = slconfig.fromfile(str(config_path))
    args.device = str(torch.device(device))
    # Build the checkpoint's untouched architecture.  In particular this must
    # remain 91 until strict loading has succeeded.
    if args.num_classes != DINO_COCO_CLASS_COUNT:
        raise RuntimeError("Official DINO Swin-L config is no longer 91-class COCO.")
    built = build_dino(args)
    if not isinstance(built, tuple) or not built or not isinstance(built[0], nn.Module):
        raise RuntimeError("Official build_dino returned an invalid model tuple.")
    dino = built[0]
    payload: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = _validate_checkpoint_payload(payload, checkpoint_path=checkpoint_path)
    # This strict call intentionally occurs before class replacement or LoRA.
    dino.load_state_dict(model_state, strict=True)
    detector = DinoCourtDetector(
        dino,
        input_mode=input_mode,
        short_side=short_side,
        max_long_side=max_long_side,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        nested_tensor_factory=nested_tensor_factory,
    )
    detector.to(torch.device(device))
    return detector


def lora_parameter_count(model: DinoCourtDetector) -> int:
    """Return the number of LoRA scalar parameters for diagnostics."""
    return sum(parameter.numel() for parameter in iter_lora_parameters(model.dino))


__all__ = [
    "COURT_CLASS_COUNT",
    "COURT_PARAMETER_COUNT",
    "DEFAULT_DINO_LORA_TARGETS",
    "DINO_COCO_CLASS_COUNT",
    "CourtScaleAxisHead",
    "DinoCourtDetector",
    "load_pretrained_dino_court_detector",
    "lora_parameter_count",
    "validate_dino_court_output",
]
