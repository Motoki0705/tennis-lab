"""Evaluate TrackNetV3/HRNet and optimize heatmap ensembles by ACC."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.ball_detection.data.datamodule import collate_ball_sequences
from src.ball_detection.data.labeled_dataset import LabeledBallDataset
from src.ball_detection.models import build_model
from src.ball_detection.models.heatmap_utils import decode_heatmap_logits
from src.ball_detection.models.third_party_loader import load_wasb_hrnet_class


def _default_games() -> list[str]:
    return [f"game{i}" for i in range(1, 8)]


def _load_ball_detection_model(
    checkpoint_path: Path,
    device: torch.device,
    fallback_model_cfg_path: Path | None = None,
) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg: dict[str, Any] = {}
    if isinstance(state, dict):
        hyper = state.get("hyper_parameters", {})
        if isinstance(hyper, dict):
            cfg_candidate = hyper.get("config", {})
            if isinstance(cfg_candidate, dict):
                cfg = cfg_candidate

    if not cfg and fallback_model_cfg_path is not None:
        cfg = {"model": OmegaConf.to_container(OmegaConf.load(fallback_model_cfg_path), resolve=True)}

    model = build_model(cfg)

    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    remapped = {
        (k.replace("model.", "", 1) if isinstance(k, str) and k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint {checkpoint_path}: {unexpected}")
    if missing:
        print(f"[warn] missing keys while loading {checkpoint_path}: {len(missing)}")

    model.to(device)
    model.eval()
    return model


def _load_wasb_hrnet_model(
    *,
    wasb_model_cfg_path: Path,
    wasb_checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    cfg = OmegaConf.load(wasb_model_cfg_path)
    hrnet_cls = load_wasb_hrnet_class()
    model = hrnet_cls(cfg)

    checkpoint = torch.load(wasb_checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported WASB checkpoint format: {wasb_checkpoint_path}")

    state_dict: dict[str, Tensor] | None = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(isinstance(v, Tensor) for v in checkpoint.values()):
        state_dict = checkpoint  # type: ignore[assignment]
    if state_dict is None:
        raise ValueError(f"No state dict found in WASB checkpoint: {wasb_checkpoint_path}")

    target_state = model.state_dict()
    loadable: dict[str, Tensor] = {}
    for raw_key, tensor in state_dict.items():
        candidates = [
            raw_key,
            str(raw_key).removeprefix("model."),
            str(raw_key).removeprefix("module."),
            str(raw_key).removeprefix("model.").removeprefix("module."),
        ]
        for candidate in candidates:
            if candidate in target_state and target_state[candidate].shape == tensor.shape:
                loadable[candidate] = tensor
                break
    if not loadable:
        raise ValueError(f"No compatible WASB weights found: {wasb_checkpoint_path}")

    merged = target_state
    merged.update(loadable)
    model.load_state_dict(merged, strict=False)
    model.to(device)
    model.eval()
    return model


def _wasb_center_logits(hrnet_model: nn.Module, frames_ctx: Tensor) -> Tensor:
    batch_size, seq_len, channels, height, width = frames_ctx.shape
    if channels != 9:
        raise ValueError(f"WASB HRNet expects context-stacked channels=9, got {channels}")

    flat = frames_ctx.reshape(batch_size * seq_len, channels, height, width)
    out = hrnet_model(flat)
    if isinstance(out, dict):
        if 0 not in out:
            raise KeyError("WASB HRNet output dict does not contain scale key 0")
        out = out[0]

    if out.dim() != 4:
        raise ValueError(f"Unexpected WASB output shape: {tuple(out.shape)}")
    if out.shape[1] == 1:
        center = out[:, 0]
    else:
        center = out[:, out.shape[1] // 2]
    return center.view(batch_size, seq_len, center.shape[-2], center.shape[-1])


def _update_counts(
    *,
    pred_xy: Tensor,
    target_xy: Tensor,
    target_vis: Tensor,
    frame_mask: Tensor,
    acc_threshold_px: float,
    image_w: int,
    image_h: int,
) -> tuple[int, int]:
    valid = (frame_mask > 0) & (target_vis > 0)
    dx = (pred_xy[..., 0] - target_xy[..., 0]) * max(image_w - 1, 1)
    dy = (pred_xy[..., 1] - target_xy[..., 1]) * max(image_h - 1, 1)
    dist = torch.sqrt(dx * dx + dy * dy)

    correct = ((dist <= acc_threshold_px) & valid).sum().item()
    total = valid.sum().item()
    return int(correct), int(total)


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _evaluate(
    *,
    dataloader: DataLoader,
    track_model: nn.Module,
    hrnet_model: nn.Module,
    device: torch.device,
    acc_threshold_px: float,
    image_w: int,
    image_h: int,
    amp_mode: str,
    amp_dtype: torch.dtype,
    log_every: int,
    show_progress: bool,
) -> dict[str, float]:
    counters: dict[str, list[int]] = {
        "tracknet": [0, 0],
        "wasb_hrnet": [0, 0],
    }

    logit_weights = [0.5, 0.7, 0.9]
    prob_weights = [0.5, 0.7, 0.9]
    select_deltas = [-1.0, 0.0, 1.0]
    coord_weights = [round(float(x), 2) for x in np.linspace(0.0, 1.0, 21)]

    for w in logit_weights:
        counters[f"ens_logit_w{w:.2f}"] = [0, 0]
    for w in prob_weights:
        counters[f"ens_prob_w{w:.2f}"] = [0, 0]
    for d in select_deltas:
        counters[f"ens_select_d{d:+.1f}"] = [0, 0]
    for w in coord_weights:
        counters[f"ens_coord_w{w:.2f}"] = [0, 0]

    eps = 1e-6

    total_batches = len(dataloader)
    start_time = time.perf_counter()

    with torch.inference_mode():
        iterator = tqdm(
            dataloader,
            total=total_batches,
            desc="Evaluating",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        for batch_idx, batch in enumerate(iterator, start=1):
            frames_ctx = batch["frames"].to(device, non_blocking=True)
            if device.type == "cuda" and amp_mode != "none":
                frames_ctx = frames_ctx.to(dtype=amp_dtype)
            target_xy = batch["target_xy"].to(device)
            target_vis = batch["target_vis"].to(device)
            frame_mask = batch["frame_mask"].to(device)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if device.type == "cuda" and amp_mode != "none"
                else nullcontext()
            )
            with amp_ctx:
                track_frames = frames_ctx[:, :, 3:6]
                track_logits = track_model(track_frames)["heatmap_logits"]
                hr_logits = _wasb_center_logits(hrnet_model, frames_ctx)

            track_logits = track_logits.float()
            hr_logits = hr_logits.float()

            track_xy, track_conf = decode_heatmap_logits(track_logits)
            hr_xy, hr_conf = decode_heatmap_logits(hr_logits)

            c, t = _update_counts(
                pred_xy=track_xy,
                target_xy=target_xy,
                target_vis=target_vis,
                frame_mask=frame_mask,
                acc_threshold_px=acc_threshold_px,
                image_w=image_w,
                image_h=image_h,
            )
            counters["tracknet"][0] += c
            counters["tracknet"][1] += t

            c, t = _update_counts(
                pred_xy=hr_xy,
                target_xy=target_xy,
                target_vis=target_vis,
                frame_mask=frame_mask,
                acc_threshold_px=acc_threshold_px,
                image_w=image_w,
                image_h=image_h,
            )
            counters["wasb_hrnet"][0] += c
            counters["wasb_hrnet"][1] += t

            for w in logit_weights:
                ens_logits = track_logits * w + hr_logits * (1.0 - w)
                ens_xy, _ = decode_heatmap_logits(ens_logits)
                c, t = _update_counts(
                    pred_xy=ens_xy,
                    target_xy=target_xy,
                    target_vis=target_vis,
                    frame_mask=frame_mask,
                    acc_threshold_px=acc_threshold_px,
                    image_w=image_w,
                    image_h=image_h,
                )
                counters[f"ens_logit_w{w:.2f}"][0] += c
                counters[f"ens_logit_w{w:.2f}"][1] += t

            track_prob = torch.sigmoid(track_logits)
            hr_prob = torch.sigmoid(hr_logits)
            for w in prob_weights:
                ens_prob = track_prob * w + hr_prob * (1.0 - w)
                ens_logits = torch.logit(torch.clamp(ens_prob, min=eps, max=1.0 - eps))
                ens_xy, _ = decode_heatmap_logits(ens_logits)
                c, t = _update_counts(
                    pred_xy=ens_xy,
                    target_xy=target_xy,
                    target_vis=target_vis,
                    frame_mask=frame_mask,
                    acc_threshold_px=acc_threshold_px,
                    image_w=image_w,
                    image_h=image_h,
                )
                counters[f"ens_prob_w{w:.2f}"][0] += c
                counters[f"ens_prob_w{w:.2f}"][1] += t

            for d in select_deltas:
                use_track = track_conf >= (hr_conf + d)
                ens_xy = torch.where(use_track.unsqueeze(-1), track_xy, hr_xy)
                c, t = _update_counts(
                    pred_xy=ens_xy,
                    target_xy=target_xy,
                    target_vis=target_vis,
                    frame_mask=frame_mask,
                    acc_threshold_px=acc_threshold_px,
                    image_w=image_w,
                    image_h=image_h,
                )
                counters[f"ens_select_d{d:+.1f}"][0] += c
                counters[f"ens_select_d{d:+.1f}"][1] += t

            for w in coord_weights:
                ens_xy = track_xy * w + hr_xy * (1.0 - w)
                c, t = _update_counts(
                    pred_xy=ens_xy,
                    target_xy=target_xy,
                    target_vis=target_vis,
                    frame_mask=frame_mask,
                    acc_threshold_px=acc_threshold_px,
                    image_w=image_w,
                    image_h=image_h,
                )
                counters[f"ens_coord_w{w:.2f}"][0] += c
                counters[f"ens_coord_w{w:.2f}"][1] += t

            if show_progress:
                elapsed = time.perf_counter() - start_time
                per_batch = elapsed / max(batch_idx, 1)
                remaining = per_batch * max(total_batches - batch_idx, 0)

                track_total = max(counters["tracknet"][1], 1)
                hr_total = max(counters["wasb_hrnet"][1], 1)
                track_acc = counters["tracknet"][0] / track_total
                hr_acc = counters["wasb_hrnet"][0] / hr_total

                iterator.set_postfix(
                    {
                        "trk": f"{track_acc:.3f}",
                        "hr": f"{hr_acc:.3f}",
                        "eta": _format_seconds(remaining),
                    }
                )

                if batch_idx % max(log_every, 1) == 0 or batch_idx == total_batches:
                    print(
                        "[eval] "
                        f"{batch_idx}/{total_batches} "
                        f"elapsed={_format_seconds(elapsed)} "
                        f"eta={_format_seconds(remaining)} "
                        f"track_acc={track_acc:.4f} hrnet_acc={hr_acc:.4f}",
                        flush=True,
                    )

    results: dict[str, float] = {}
    for name, (correct, total) in counters.items():
        results[name] = float(correct) / float(max(total, 1))
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heatmap ensemble ACC.")
    parser.add_argument(
        "--tracknet-checkpoint",
        type=Path,
        required=True,
        help="TrackNetV3 lightning checkpoint path (.ckpt).",
    )
    parser.add_argument(
        "--tracknet-model-config",
        type=Path,
        default=Path("src/ball_detection/configs/model/tracknetv3.yaml"),
        help="Fallback TrackNetV3 model config when checkpoint lacks config.",
    )
    parser.add_argument(
        "--wasb-checkpoint",
        type=Path,
        default=Path("checkpoints/wasb/wasb_tennis_best.pth.tar"),
        help="WASB HRNet pretrained checkpoint path.",
    )
    parser.add_argument(
        "--wasb-model-config",
        type=Path,
        default=Path("third_party/WASB-SBDT/src/configs/model/wasb.yaml"),
        help="WASB model yaml path.",
    )
    parser.add_argument("--root-dir", type=Path, default=Path("data/tennis"))
    parser.add_argument("--games", nargs="+", default=_default_games())
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--image-h", type=int, default=288)
    parser.add_argument("--image-w", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--acc-threshold-px", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    amp_mode = str(args.amp)
    if device.type != "cuda":
        amp_mode = "none"

    amp_dtype = torch.float16
    if amp_mode == "bf16":
        amp_dtype = torch.bfloat16

    print(f"Using device: {device}")
    print(
        "Eval settings: "
        f"batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"prefetch_factor={args.prefetch_factor}, pin_memory={bool(args.pin_memory)}, "
        f"persistent_workers={bool(args.persistent_workers)}, amp={amp_mode}",
        flush=True,
    )

    dataset = LabeledBallDataset(
        root_dir=args.root_dir,
        games=list(args.games),
        image_size_hw=(int(args.image_h), int(args.image_w)),
        window_size=int(args.seq_len),
        window_stride=int(args.seq_len),
        min_window_size=int(args.seq_len),
        context_frames=3,
    )
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "drop_last": False,
        "collate_fn": collate_ball_sequences,
    }
    if int(args.num_workers) > 0:
        dataloader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        dataloader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    track_model = _load_ball_detection_model(
        checkpoint_path=args.tracknet_checkpoint,
        device=device,
        fallback_model_cfg_path=args.tracknet_model_config,
    )
    hrnet_model = _load_wasb_hrnet_model(
        wasb_model_cfg_path=args.wasb_model_config,
        wasb_checkpoint_path=args.wasb_checkpoint,
        device=device,
    )

    results = _evaluate(
        dataloader=dataloader,
        track_model=track_model,
        hrnet_model=hrnet_model,
        device=device,
        acc_threshold_px=float(args.acc_threshold_px),
        image_w=int(args.image_w),
        image_h=int(args.image_h),
        amp_mode=amp_mode,
        amp_dtype=amp_dtype,
        log_every=int(args.log_every),
        show_progress=not bool(args.no_progress),
    )

    single_best = max(results.get("tracknet", 0.0), results.get("wasb_hrnet", 0.0))
    ensemble_items = [(k, v) for k, v in results.items() if k not in {"tracknet", "wasb_hrnet"}]
    ensemble_best_name, ensemble_best_acc = max(ensemble_items, key=lambda kv: kv[1])

    print("\n=== ACC Summary ===")
    print(f"tracknet: {results['tracknet']:.6f}")
    print(f"wasb_hrnet: {results['wasb_hrnet']:.6f}")
    print(f"best_ensemble: {ensemble_best_name} -> {ensemble_best_acc:.6f}")
    print(f"ensemble_better_than_both: {ensemble_best_acc > single_best}")

    top10 = sorted(ensemble_items, key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop ensemble methods:")
    for name, value in top10:
        print(f"  {name}: {value:.6f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tracknet": results["tracknet"],
            "wasb_hrnet": results["wasb_hrnet"],
            "best_ensemble": {"name": ensemble_best_name, "acc": ensemble_best_acc},
            "ensemble_better_than_both": bool(ensemble_best_acc > single_best),
            "all": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved json report: {args.output_json}")


if __name__ == "__main__":
    main()
