#!/usr/bin/env python3
"""Load the Swin backbone from a full DINO checkpoint.

This script mirrors the Swin backbone path defined in `third_party/DINO` and targets
full DINO checkpoints whose model weights contain keys like
`backbone.0.patch_embed.proj.weight`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY_DINO_ROOT = ROOT / "third_party" / "DINO"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(THIRD_PARTY_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DINO_ROOT))

import torch
from torch import nn


DEFAULT_WEIGHTS_ROOT = Path(
    os.environ.get(
        "MODEL_WEIGHTS_ROOT",
        "/mnt/d/weights" if Path("/mnt/d/weights").exists() else "D:/weights",
    )
)
DINO_WEIGHTS_DIR = DEFAULT_WEIGHTS_ROOT / "DINO"


def load_swin_builder():
    """Load third_party/DINO's Swin builder without importing the full DINO package."""

    module_path = THIRD_PARTY_DINO_ROOT / "models" / "dino" / "swin_transformer.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Swin transformer builder not found at {module_path}. "
            "Initialize third_party/DINO before using Swin backbones."
        )
    module_name = "third_party_dino_swin_transformer"
    if module_name in sys.modules:
        return sys.modules[module_name].build_swin_transformer
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.build_swin_transformer


build_swin_transformer = load_swin_builder()


BACKBONE_CHOICES = [
    "swin_T_224_1k",
    "swin_B_224_22k",
    "swin_B_384_22k",
    "swin_L_224_22k",
    "swin_L_384_22k",
]


class DINOSwinBackbone(nn.Module):
    """Thin wrapper that exposes Swin raw features as an OrderedDict."""

    def __init__(
        self,
        *,
        backbone: str = "swin_L_384_22k",
        pretrain_img_size: int = 384,
        dilation: bool = False,
        return_interm_indices: list[int] | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if return_interm_indices is None:
            return_interm_indices = [0, 1, 2, 3]
        self.backbone = build_swin_transformer(
            backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=dilation,
            use_checkpoint=use_checkpoint,
        )
        self.backbone_name = backbone
        self.pretrain_img_size = pretrain_img_size
        self.return_interm_indices = return_interm_indices
        self.dilation = dilation
        self.use_checkpoint = use_checkpoint

    def forward_raw(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.backbone.forward_raw(x)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        outputs = self.forward_raw(x)
        keys = [str(index) for index in self.return_interm_indices]
        return OrderedDict(zip(keys, outputs, strict=True))


def checkpoint_args_to_dict(args: Any) -> dict[str, Any]:
    """Convert checkpoint args object into a JSON-serializable dict."""
    if args is None:
        return {}
    if isinstance(args, dict):
        raw = args
    elif hasattr(args, "__dict__"):
        raw = vars(args)
    else:
        raise TypeError(f"Unsupported args type: {type(args)!r}")
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result


def infer_pretrain_img_size(backbone: str) -> int:
    return int(backbone.split("_")[-2])


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(checkpoint)!r}")
    return checkpoint


def get_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Expected state dict, got {type(state)!r}")
    return state


def extract_backbone_state(
    state_dict: dict[str, torch.Tensor],
    prefix: str = "backbone.0.",
) -> OrderedDict[str, torch.Tensor]:
    extracted = OrderedDict(
        (key[len(prefix) :], value)
        for key, value in state_dict.items()
        if key.startswith(prefix)
    )
    if not extracted:
        raise KeyError(f"No keys found with prefix {prefix!r}")
    return extracted


def summarize_key_prefixes(state_dict: dict[str, torch.Tensor], depth: int = 3) -> dict[str, int]:
    counter = Counter()
    for key in state_dict:
        prefix = ".".join(key.split(".")[:depth])
        counter[prefix] += 1
    return dict(counter.most_common())


def build_metadata_namespace(
    *,
    backbone: str,
    pretrain_img_size: int,
    dilation: bool,
    return_interm_indices: list[int],
    use_checkpoint: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        backbone=backbone,
        pretrain_img_size=pretrain_img_size,
        dilation=dilation,
        return_interm_indices=return_interm_indices,
        use_checkpoint=use_checkpoint,
    )


def resolve_backbone_config(
    checkpoint: dict[str, Any],
    *,
    backbone: str | None,
    pretrain_img_size: int | None,
    dilation: bool | None,
    return_interm_indices: list[int] | None,
    use_checkpoint: bool | None,
) -> SimpleNamespace:
    args_dict = checkpoint_args_to_dict(checkpoint.get("args"))
    resolved_backbone = backbone or args_dict.get("backbone") or "swin_L_384_22k"
    if resolved_backbone not in BACKBONE_CHOICES:
        raise ValueError(f"Unsupported backbone: {resolved_backbone}")
    resolved_pretrain_img_size = (
        pretrain_img_size
        if pretrain_img_size is not None
        else infer_pretrain_img_size(resolved_backbone)
    )
    resolved_dilation = dilation if dilation is not None else bool(args_dict.get("dilation", False))
    resolved_return_interm = (
        return_interm_indices
        if return_interm_indices is not None
        else list(args_dict.get("return_interm_indices", [0, 1, 2, 3]))
    )
    resolved_use_checkpoint = (
        use_checkpoint if use_checkpoint is not None else bool(args_dict.get("use_checkpoint", False))
    )
    return build_metadata_namespace(
        backbone=resolved_backbone,
        pretrain_img_size=resolved_pretrain_img_size,
        dilation=resolved_dilation,
        return_interm_indices=resolved_return_interm,
        use_checkpoint=resolved_use_checkpoint,
    )


def load_swin_backbone_state(
    checkpoint_path: str | Path,
    *,
    backbone: str = "swin_L_384_22k",
    pretrain_img_size: int = 384,
    dilation: bool = False,
    return_interm_indices: list[int] | None = None,
    use_checkpoint: bool = False,
    strict: bool = True,
) -> tuple[DINOSwinBackbone, nn.modules.module._IncompatibleKeys]:
    """Load a trimmed Swin backbone checkpoint into ``DINOSwinBackbone``."""

    model = DINOSwinBackbone(
        backbone=backbone,
        pretrain_img_size=pretrain_img_size,
        dilation=dilation,
        return_interm_indices=return_interm_indices,
        use_checkpoint=use_checkpoint,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_result = model.backbone.load_state_dict(state_dict, strict=strict)
    return model, load_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DINO_WEIGHTS_DIR / "checkpoint0027_5scale_swin.pth",
        help="Path to the full DINO checkpoint.",
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONE_CHOICES,
        help="Backbone architecture. Defaults to checkpoint metadata when available.",
    )
    parser.add_argument(
        "--pretrain-img-size",
        type=int,
        help="Pretraining image size. Defaults to the backbone name suffix.",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        default=None,
        help="Enable dilation in the last Swin stage.",
    )
    parser.add_argument(
        "--no-dilation",
        action="store_false",
        dest="dilation",
        help="Disable dilation in the last Swin stage.",
    )
    parser.add_argument(
        "--return-interm-indices",
        type=int,
        nargs="+",
        help="Intermediate Swin stages returned by the backbone wrapper.",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        default=None,
        help="Enable Swin activation checkpointing.",
    )
    parser.add_argument(
        "--no-use-checkpoint",
        action="store_false",
        dest="use_checkpoint",
        help="Disable Swin activation checkpointing.",
    )
    parser.add_argument(
        "--save-backbone-state",
        type=Path,
        help="Optional output path for the trimmed Swin backbone state dict.",
    )
    parser.add_argument(
        "--save-full-backbone-module",
        type=Path,
        help="Optional output path for the loaded DINOSwinBackbone state dict.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading the extracted backbone weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    state_dict = get_model_state(checkpoint)
    backbone_state = extract_backbone_state(state_dict)
    resolved = resolve_backbone_config(
        checkpoint,
        backbone=args.backbone,
        pretrain_img_size=args.pretrain_img_size,
        dilation=args.dilation,
        return_interm_indices=args.return_interm_indices,
        use_checkpoint=args.use_checkpoint,
    )

    print("Checkpoint analysis")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  top_level_keys: {list(checkpoint.keys())}")
    print(f"  total model keys: {len(state_dict)}")
    print(f"  extracted backbone keys: {len(backbone_state)}")
    print(f"  backbone: {resolved.backbone}")
    print(f"  pretrain_img_size: {resolved.pretrain_img_size}")
    print(f"  dilation: {resolved.dilation}")
    print(f"  return_interm_indices: {resolved.return_interm_indices}")
    print(f"  use_checkpoint: {resolved.use_checkpoint}")
    print("  model prefix counts:")
    for prefix, count in summarize_key_prefixes(state_dict).items():
        print(f"    {prefix}: {count}")
    print("  sample extracted keys:")
    for key in list(backbone_state.keys())[:10]:
        print(f"    {key}")

    if args.save_backbone_state is not None:
        args.save_backbone_state.parent.mkdir(parents=True, exist_ok=True)
        torch.save(backbone_state, args.save_backbone_state)
        print(f"Saved trimmed backbone state to: {args.save_backbone_state}")
        load_path = args.save_backbone_state
    else:
        load_path = args.checkpoint.with_suffix(".swin_backbone.tmp.pth")
        torch.save(backbone_state, load_path)

    model, load_result = load_swin_backbone_state(
        load_path,
        backbone=resolved.backbone,
        pretrain_img_size=resolved.pretrain_img_size,
        dilation=resolved.dilation,
        return_interm_indices=resolved.return_interm_indices,
        use_checkpoint=resolved.use_checkpoint,
        strict=args.strict,
    )

    print("Load result")
    print(f"  load_source: {load_path}")
    print(f"  missing_keys: {list(load_result.missing_keys)}")
    print(f"  unexpected_keys: {list(load_result.unexpected_keys)}")

    if args.save_full_backbone_module is not None:
        args.save_full_backbone_module.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_full_backbone_module)
        print(f"Saved loaded DINOSwinBackbone state to: {args.save_full_backbone_module}")

    if args.save_backbone_state is None and load_path.exists():
        load_path.unlink()

    summary = {
        "checkpoint": str(args.checkpoint),
        "top_level_keys": list(checkpoint.keys()),
        "resolved_config": {
            "backbone": resolved.backbone,
            "pretrain_img_size": resolved.pretrain_img_size,
            "dilation": resolved.dilation,
            "return_interm_indices": resolved.return_interm_indices,
            "use_checkpoint": resolved.use_checkpoint,
        },
        "load_result": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
    }
    print("Resolved summary json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
