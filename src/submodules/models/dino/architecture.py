"""Official IDEA-Research DINO 4-scale Swin-L architecture shared by all callers.

Inference (:mod:`person_detector`) and fine-tuning
(:mod:`src.tasks.player_detection`) build the model through this module so both
use exactly the same upstream code, configuration, preprocessing, and checkpoint
contract. The upstream source is imported from the pinned git submodule without
copying or modifying it.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision.transforms import functional as transform_functional

# The released COCO checkpoint predicts person at class id 1. Fine-tuned
# player checkpoints keep the 91-way head and reuse this id for "player".
COCO_PERSON_CLASS_ID = 1
DINO_NUM_CLASSES = 91
DINO_SWIN_BACKBONE = "swin_L_384_22k"
DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGE_STD = (0.229, 0.224, 0.225)
_DINO_MODELS_PACKAGE = "_tennis_lab_third_party_dino_models"


class DinoConfig(SimpleNamespace):
    """Attribute config with the membership behavior expected upstream."""

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and hasattr(self, name)


def dino_4scale_swin_args(
    device: torch.device | str, *, use_checkpoint: bool
) -> DinoConfig:
    """Exact official ``DINO_4scale_swin.py`` configuration.

    ``use_checkpoint`` enables Swin activation checkpointing for training; it
    changes memory use only, never the parameters or outputs.
    """
    return DinoConfig(
        device=str(device),
        num_classes=DINO_NUM_CLASSES,
        backbone=DINO_SWIN_BACKBONE,
        lr_backbone=1e-5,
        dilation=False,
        return_interm_indices=[1, 2, 3],
        backbone_freeze_keywords=None,
        use_checkpoint=use_checkpoint,
        position_embedding="sine",
        pe_temperatureH=20,
        pe_temperatureW=20,
        enc_layers=6,
        dec_layers=6,
        unic_layers=0,
        pre_norm=False,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.0,
        nheads=8,
        num_queries=900,
        query_dim=4,
        num_patterns=0,
        num_feature_levels=4,
        enc_n_points=4,
        dec_n_points=4,
        decoder_layer_noise=False,
        dln_xy_noise=0.2,
        dln_hw_noise=0.2,
        decoder_module_seq=["sa", "ca", "ffn"],
        decoder_sa_type="sa",
        dec_layer_number=None,
        transformer_activation="relu",
        use_deformable_box_attn=False,
        box_attn_type="roi_align",
        add_channel_attention=False,
        add_pos_value=False,
        random_refpoints_xy=False,
        fix_refpoints_hw=-1,
        two_stage_type="standard",
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_bbox_embed_share=False,
        two_stage_class_embed_share=False,
        two_stage_learn_wh=False,
        two_stage_keep_all_tokens=False,
        dec_pred_bbox_embed_share=True,
        dec_pred_class_embed_share=True,
        use_detached_boxes_dec_out=False,
        use_dn=True,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=DINO_NUM_CLASSES,
        embed_init_tgt=True,
        masks=False,
        frozen_weights=None,
        aux_loss=True,
        matcher_type="HungarianMatcher",
        set_cost_class=2.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        cls_loss_coef=1.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        mask_loss_coef=1.0,
        dice_loss_coef=1.0,
        focal_alpha=0.25,
        no_interm_box_loss=False,
        interm_loss_coef=1.0,
        match_unstable_error=True,
        num_select=300,
        nms_iou_threshold=-1,
    )


def build_dino(
    repository: Path, args: DinoConfig
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Build the official model and its training criterion (unloaded weights)."""
    try:
        build_function = load_dino_build_function(repository)
    except ModuleNotFoundError as error:
        if error.name == "MultiScaleDeformableAttention":
            raise RuntimeError(
                "DINO CUDA extension is not installed. Run: "
                "TENNIS_LAB_BUILD_CUDA_OPS=1 "
                ".venv/bin/python setup.py build_ext --inplace"
            ) from error
        raise
    model, criterion, _ = build_function(args)
    if not isinstance(model, torch.nn.Module) or not isinstance(
        criterion, torch.nn.Module
    ):
        raise TypeError("Official build_dino did not return torch modules.")
    return model, criterion


def load_dino_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Read a DINO-format checkpoint (``{"model", "args"}``) after validation."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINO checkpoint not found: {checkpoint}")
    payload: Any = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload or "args" not in payload:
        raise ValueError("DINO checkpoint must contain both 'model' and 'args' entries")
    validate_checkpoint_args(payload["args"])
    state_dict = payload["model"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("DINO checkpoint 'model' entry must be a state-dict mapping")
    return dict(state_dict)


def load_dino_build_function(repository: Path) -> Callable[[Any], tuple[Any, ...]]:
    """Load official DINO without modifying or copying its source tree."""
    models_init, util_init = validate_dino_repository(repository)
    _load_dino_util_package(util_init)

    module = sys.modules.get(_DINO_MODELS_PACKAGE)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            _DINO_MODELS_PACKAGE,
            models_init,
            submodule_search_locations=[str(models_init.parent)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create an import spec for {models_init}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_DINO_MODELS_PACKAGE] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            for name in tuple(sys.modules):
                if name == _DINO_MODELS_PACKAGE or name.startswith(
                    f"{_DINO_MODELS_PACKAGE}."
                ):
                    sys.modules.pop(name, None)
            raise
    else:
        loaded_file = getattr(module, "__file__", None)
        if loaded_file is None or Path(loaded_file).resolve() != models_init.resolve():
            raise RuntimeError(
                "DINO models are already imported from a different repository: "
                f"{loaded_file!r}"
            )

    build_dino_function = getattr(module, "build_dino", None)
    if not callable(build_dino_function):
        raise RuntimeError(
            f"Official DINO module has no build_dino function: {models_init}"
        )
    return cast(Callable[[Any], tuple[Any, ...]], build_dino_function)


def validate_dino_repository(repository: Path) -> tuple[Path, Path]:
    models_init = repository / "models/__init__.py"
    util_init = repository / "util/__init__.py"
    missing = [path for path in (models_init, util_init) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "DINO git submodule is not initialized. Run: "
            "git submodule update --init third_party/DINO "
            f"(missing: {', '.join(str(path) for path in missing)})"
        )
    return models_init, util_init


def _load_dino_util_package(util_init: Path) -> None:
    """Expose DINO's upstream absolute ``util.*`` imports explicitly."""
    loaded = sys.modules.get("util")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is None or Path(loaded_file).resolve() != util_init.resolve():
            raise RuntimeError(
                "Cannot load DINO because another top-level 'util' package is already "
                f"imported from {loaded_file!r}"
            )
        return

    spec = importlib.util.spec_from_file_location(
        "util",
        util_init,
        submodule_search_locations=[str(util_init.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create an import spec for {util_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["util"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("util", None)
        raise


def validate_checkpoint_args(args: Any) -> None:
    # Released checkpoints store CLI args but omit config-inherited fields.
    # The remaining architecture is checked by strict state-dict loading.
    expected = {"backbone": DINO_SWIN_BACKBONE}
    mismatches = {
        name: (expected_value, getattr(args, name, None))
        for name, expected_value in expected.items()
        if getattr(args, name, None) != expected_value
    }
    if mismatches:
        details = ", ".join(
            f"{name}: expected {wanted!r}, got {actual!r}"
            for name, (wanted, actual) in mismatches.items()
        )
        raise ValueError(f"Unsupported DINO checkpoint architecture ({details})")


def dino_resized_shape(
    height: int, width: int, *, short_side: int, max_long_side: int
) -> tuple[int, int]:
    """Official DINO test resize: short side target capped by the long side."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Image size must be positive, got {(height, width)}")
    if short_side <= 0 or max_long_side < short_side:
        raise ValueError(
            "Expected 0 < short_side <= max_long_side, got "
            f"{short_side} and {max_long_side}"
        )
    scale = short_side / min(height, width)
    if max(height, width) * scale > max_long_side:
        scale = max_long_side / max(height, width)
    return int(round(height * scale)), int(round(width * scale))


def preprocess_frame(
    frame_bgr: NDArray[np.uint8], *, short_side: int, max_long_side: int
) -> torch.Tensor:
    """BGR uint8 frame -> resized, ImageNet-normalized RGB ``(3,H,W)`` float."""
    image = torch.from_numpy(np.ascontiguousarray(frame_bgr[..., ::-1])).permute(
        2, 0, 1
    )
    image = image.float().div_(255.0)
    height, width = image.shape[-2:]
    target_size = dino_resized_shape(
        int(height), int(width), short_side=short_side, max_long_side=max_long_side
    )
    image = transform_functional.resize(image, list(target_size), antialias=True)
    return cast(
        torch.Tensor,
        transform_functional.normalize(
            image, mean=list(DINO_IMAGE_MEAN), std=list(DINO_IMAGE_STD)
        ),
    )
