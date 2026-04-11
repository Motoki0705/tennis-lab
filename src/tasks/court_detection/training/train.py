"""Training loops for court detection tasks.

Three training functions:

* :func:`train_seg` — Court cell segmentation (CE + Dice).
* :func:`train_kp` — Court keypoint heatmap regression (Focal BCE).
* :func:`train_line` — Court white-line segmentation (BCE + Dice).
"""

from __future__ import annotations

import csv
import dataclasses
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.tasks.court_detection.models.court_unet import CourtUNet
from src.tasks.court_detection.training.losses import (
    BinaryDiceLoss,
    DiceLoss,
    FocalBCEWithLogitsLoss,
)
from src.tasks.court_detection.training.visualization import (
    save_kp_vis,
    save_line_vis,
    save_seg_vis,
)

# ── Custom Collate (variable-size images) ─────────────────────────


def _pad_collate_seg(batch: list[dict]) -> dict:
    """Pad variable-size images/masks to the max size in the batch."""
    max_h = max(b["image"].shape[1] for b in batch)
    max_w = max(b["image"].shape[2] for b in batch)
    max_h = ((max_h + 7) // 8) * 8
    max_w = ((max_w + 7) // 8) * 8

    images = []
    masks = []
    ids = []
    for b in batch:
        c, h, w = b["image"].shape
        padded_img = torch.zeros(c, max_h, max_w, dtype=b["image"].dtype)
        padded_img[:, :h, :w] = b["image"]
        images.append(padded_img)

        padded_mask = torch.zeros(max_h, max_w, dtype=b["mask"].dtype)
        padded_mask[:h, :w] = b["mask"]
        masks.append(padded_mask)

        ids.append(b["image_id"])

    return {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
        "image_id": ids,
    }


def _pad_collate_line(batch: list[dict]) -> dict:
    """Pad variable-size images/binary masks to the max size in the batch."""
    max_h = max(b["image"].shape[1] for b in batch)
    max_w = max(b["image"].shape[2] for b in batch)
    max_h = ((max_h + 7) // 8) * 8
    max_w = ((max_w + 7) // 8) * 8

    images = []
    masks = []
    ids = []
    for b in batch:
        c, h, w = b["image"].shape
        padded_img = torch.zeros(c, max_h, max_w, dtype=b["image"].dtype)
        padded_img[:, :h, :w] = b["image"]
        images.append(padded_img)

        _, mh, mw = b["mask"].shape
        padded_mask = torch.zeros(1, max_h, max_w, dtype=b["mask"].dtype)
        padded_mask[:, :mh, :mw] = b["mask"]
        masks.append(padded_mask)
        ids.append(b["image_id"])

    return {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
        "image_id": ids,
    }


def _pad_collate_kp(batch: list[dict]) -> dict:
    """Pad variable-size images/heatmaps to the max size in the batch."""
    max_h = max(b["image"].shape[1] for b in batch)
    max_w = max(b["image"].shape[2] for b in batch)
    max_h = ((max_h + 7) // 8) * 8
    max_w = ((max_w + 7) // 8) * 8

    images = []
    heatmaps = []
    keypoints = []
    ids = []
    for b in batch:
        c, h, w = b["image"].shape
        padded_img = torch.zeros(c, max_h, max_w, dtype=b["image"].dtype)
        padded_img[:, :h, :w] = b["image"]
        images.append(padded_img)

        n, hh, hw = b["heatmap"].shape
        padded_hm = torch.zeros(n, max_h, max_w, dtype=b["heatmap"].dtype)
        padded_hm[:, :hh, :hw] = b["heatmap"]
        heatmaps.append(padded_hm)

        keypoints.append(b["keypoints"])
        ids.append(b["image_id"])

    return {
        "image": torch.stack(images),
        "heatmap": torch.stack(heatmaps),
        "keypoints": torch.stack(keypoints),
        "image_id": ids,
    }


# ── Training Loops ────────────────────────────────────────────────


def train_seg(
    *,
    dry_run: bool = False,
    resume: str | None = None,
    save_train_vis: bool = True,
    save_vis_every: int = 5,
    save_vis_max_samples: int = 8,
) -> dict[str, Any]:
    """Run court segmentation training."""
    from src.tasks.court_detection.configs.seg import (
        CHECKPOINT_DIR,
        COURT_DATA_DIR,
        EXPERIMENT_DIR,
        CourtSegConfig,
    )
    from src.tasks.court_detection.data.court_seg_dataset import CourtSegDataset

    cfg = CourtSegConfig()
    if dry_run:
        cfg.num_epochs = cfg.dry_run_epochs
        cfg.batch_size = cfg.dry_run_batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_seg] Device: {device}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cfg_dict = dataclasses.asdict(cfg)
    train_ds = CourtSegDataset(COURT_DATA_DIR, split="train", is_train=True, config=cfg_dict)
    val_ds = CourtSegDataset(COURT_DATA_DIR, split="val", is_train=False, config=cfg_dict)

    print(f"[train_seg] Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=_pad_collate_seg,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=_pad_collate_seg,
    )

    model = CourtUNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_seg] Model params: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    ce_loss_fn = nn.CrossEntropyLoss()
    dice_loss_fn = DiceLoss(num_classes=cfg.num_classes)

    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[train_seg] Resumed from epoch {start_epoch}")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_ckpt_dir = EXPERIMENT_DIR / "checkpoints"
    experiment_ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = EXPERIMENT_DIR / "metrics.csv"

    write_header = not metrics_file.exists() or start_epoch == 0
    metrics_fp = open(metrics_file, "a", newline="")  # noqa: SIM115
    csv_writer = csv.writer(metrics_fp)
    if write_header:
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "time_s"])

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()

        model.train()
        running_loss = 0.0
        vis_saved = 0
        for batch in tqdm(train_dl, desc=f"[seg] Train E{epoch}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss_ce = ce_loss_fn(logits, masks)
            loss_dice = dice_loss_fn(logits, masks)
            loss = cfg.ce_weight * loss_ce + cfg.dice_weight * loss_dice

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
            optimizer.step()

            running_loss += loss.item()

            if save_train_vis and (epoch % save_vis_every == 0) and vis_saved < save_vis_max_samples:
                vis_dir = EXPERIMENT_DIR / "vis" / f"epoch_{epoch}"
                with torch.no_grad():
                    for b_i in range(min(images.size(0), save_vis_max_samples - vis_saved)):
                        save_seg_vis(
                            images[b_i], masks[b_i], logits[b_i],
                            vis_dir / f"train_{vis_saved}.png",
                        )
                        vis_saved += 1

        train_loss = running_loss / max(len(train_dl), 1)

        model.eval()
        val_running = 0.0
        vis_saved_val = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"[seg] Val E{epoch}"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                loss_ce = ce_loss_fn(logits, masks)
                loss_dice = dice_loss_fn(logits, masks)
                loss = cfg.ce_weight * loss_ce + cfg.dice_weight * loss_dice
                val_running += loss.item()

                if (epoch % save_vis_every == 0) and vis_saved_val < save_vis_max_samples:
                    vis_dir = EXPERIMENT_DIR / "vis" / f"epoch_{epoch}"
                    for b_i in range(min(images.size(0), save_vis_max_samples - vis_saved_val)):
                        save_seg_vis(
                            images[b_i], masks[b_i], logits[b_i],
                            vis_dir / f"val_{vis_saved_val}.png",
                        )
                        vis_saved_val += 1

        val_loss = val_running / max(len(val_dl), 1)

        scheduler.step()
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"[seg] E{epoch} train={train_loss:.4f} val={val_loss:.4f} lr={lr:.2e} t={elapsed:.1f}s")

        csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.6e}", f"{elapsed:.1f}"])
        metrics_fp.flush()

        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss),
            "config": cfg_dict,
        }
        torch.save(ckpt_state, CHECKPOINT_DIR / "last.pt")
        torch.save(ckpt_state, experiment_ckpt_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_state, CHECKPOINT_DIR / "best.pt")
            torch.save(ckpt_state, experiment_ckpt_dir / "best.pt")
            print(f"  ★ New best val_loss: {best_val_loss:.4f}")

    metrics_fp.close()
    return {"best_val_loss": best_val_loss}


def train_kp(
    *,
    dry_run: bool = False,
    resume: str | None = None,
    save_train_vis: bool = True,
    save_vis_every: int = 5,
    save_vis_max_samples: int = 8,
) -> dict[str, Any]:
    """Run court keypoint heatmap training."""
    from src.tasks.court_detection.configs.kp import (
        CHECKPOINT_DIR,
        COURT_DATA_DIR,
        EXPERIMENT_DIR,
        CourtKPConfig,
    )
    from src.tasks.court_detection.data.court_kp_dataset import CourtKPDataset

    cfg = CourtKPConfig()
    if dry_run:
        cfg.num_epochs = cfg.dry_run_epochs
        cfg.batch_size = cfg.dry_run_batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_kp] Device: {device}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cfg_dict = dataclasses.asdict(cfg)
    train_ds = CourtKPDataset(COURT_DATA_DIR, split="train", is_train=True, config=cfg_dict)
    val_ds = CourtKPDataset(COURT_DATA_DIR, split="val", is_train=False, config=cfg_dict)

    print(f"[train_kp] Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=_pad_collate_kp,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=_pad_collate_kp,
    )

    model = CourtUNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_kp] Model params: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    loss_fn = FocalBCEWithLogitsLoss(gamma=cfg.focal_gamma)

    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[train_kp] Resumed from epoch {start_epoch}")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_ckpt_dir = EXPERIMENT_DIR / "checkpoints"
    experiment_ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = EXPERIMENT_DIR / "metrics.csv"

    write_header = not metrics_file.exists() or start_epoch == 0
    metrics_fp = open(metrics_file, "a", newline="")  # noqa: SIM115
    csv_writer = csv.writer(metrics_fp)
    if write_header:
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "time_s"])

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()

        model.train()
        running_loss = 0.0
        vis_saved = 0
        for batch in tqdm(train_dl, desc=f"[kp] Train E{epoch}"):
            images = batch["image"].to(device)
            heatmaps = batch["heatmap"].to(device)

            logits = model(images)
            loss = loss_fn(logits, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
            optimizer.step()

            running_loss += loss.item()

            if save_train_vis and (epoch % save_vis_every == 0) and vis_saved < save_vis_max_samples:
                vis_dir = EXPERIMENT_DIR / "vis" / f"epoch_{epoch}"
                with torch.no_grad():
                    for b_i in range(min(images.size(0), save_vis_max_samples - vis_saved)):
                        save_kp_vis(
                            images[b_i], heatmaps[b_i], logits[b_i],
                            vis_dir / f"train_{vis_saved}.png",
                        )
                        vis_saved += 1

        train_loss = running_loss / max(len(train_dl), 1)

        model.eval()
        val_running = 0.0
        vis_saved_val = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"[kp] Val E{epoch}"):
                images = batch["image"].to(device)
                heatmaps = batch["heatmap"].to(device)
                logits = model(images)
                loss = loss_fn(logits, heatmaps)
                val_running += loss.item()

                if (epoch % save_vis_every == 0) and vis_saved_val < save_vis_max_samples:
                    vis_dir = EXPERIMENT_DIR / "vis" / f"epoch_{epoch}"
                    for b_i in range(min(images.size(0), save_vis_max_samples - vis_saved_val)):
                        save_kp_vis(
                            images[b_i], heatmaps[b_i], logits[b_i],
                            vis_dir / f"val_{vis_saved_val}.png",
                        )
                        vis_saved_val += 1

        val_loss = val_running / max(len(val_dl), 1)

        scheduler.step()
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"[kp] E{epoch} train={train_loss:.4f} val={val_loss:.4f} lr={lr:.2e} t={elapsed:.1f}s")

        csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.6e}", f"{elapsed:.1f}"])
        metrics_fp.flush()

        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss),
            "config": cfg_dict,
        }
        torch.save(ckpt_state, CHECKPOINT_DIR / "last.pt")
        torch.save(ckpt_state, experiment_ckpt_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_state, CHECKPOINT_DIR / "best.pt")
            torch.save(ckpt_state, experiment_ckpt_dir / "best.pt")
            print(f"  ★ New best val_loss: {best_val_loss:.4f}")

    metrics_fp.close()
    return {"best_val_loss": best_val_loss}


def train_line(
    *,
    dry_run: bool = False,
    resume: str | None = None,
    save_train_vis: bool = True,
    save_vis_every: int = 5,
    save_vis_max_samples: int = 8,
) -> dict[str, Any]:
    """Run court white-line segmentation training."""
    from src.tasks.court_detection.configs.line import (
        CHECKPOINT_DIR,
        COURT_DATA_DIR,
        EXPERIMENT_DIR,
        CourtLineConfig,
    )
    from src.tasks.court_detection.data.court_line_dataset import CourtLineDataset

    cfg = CourtLineConfig()
    if dry_run:
        cfg.num_epochs = cfg.dry_run_epochs
        cfg.batch_size = cfg.dry_run_batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_line] Device: {device}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cfg_dict = dataclasses.asdict(cfg)
    train_ds = CourtLineDataset(COURT_DATA_DIR, split="train", is_train=True, config=cfg_dict)
    val_ds = CourtLineDataset(COURT_DATA_DIR, split="val", is_train=False, config=cfg_dict)

    print(f"[train_line] Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=_pad_collate_line,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=_pad_collate_line,
    )

    model = CourtUNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_line] Model params: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    pos_weight = torch.tensor([cfg.pos_weight], device=device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice_loss_fn = BinaryDiceLoss()

    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[train_line] Resumed from epoch {start_epoch}")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_ckpt_dir = EXPERIMENT_DIR / "checkpoints"
    experiment_ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = EXPERIMENT_DIR / "metrics.csv"

    write_header = not metrics_file.exists() or start_epoch == 0
    metrics_fp = open(metrics_file, "a", newline="")  # noqa: SIM115
    csv_writer = csv.writer(metrics_fp)
    if write_header:
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "time_s"])

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()

        model.train()
        running_loss = 0.0
        vis_saved = 0
        for batch in tqdm(train_dl, desc=f"[line] Train E{epoch}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss_bce = bce_loss_fn(logits, masks)
            loss_dice = dice_loss_fn(logits, masks)
            loss = cfg.bce_weight * loss_bce + cfg.dice_weight * loss_dice

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
            optimizer.step()

            running_loss += loss.item()

            if save_train_vis and (epoch % save_vis_every == 0) and vis_saved < save_vis_max_samples:
                vis_dir = EXPERIMENT_DIR / "vis" / f"epoch_{epoch}"
                with torch.no_grad():
                    for b_i in range(min(images.size(0), save_vis_max_samples - vis_saved)):
                        save_line_vis(
                            images[b_i], masks[b_i], logits[b_i],
                            vis_dir / f"train_{vis_saved}.png",
                        )
                        vis_saved += 1

        train_loss = running_loss / max(len(train_dl), 1)

        model.eval()
        val_running = 0.0
        vis_saved_val = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"[line] Val E{epoch}"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                loss_bce = bce_loss_fn(logits, masks)
                loss_dice = dice_loss_fn(logits, masks)
                loss = cfg.bce_weight * loss_bce + cfg.dice_weight * loss_dice
                val_running += loss.item()

                if (epoch % save_vis_every == 0) and vis_saved_val < save_vis_max_samples:
                    vis_dir = EXPERIMENT_DIR / "vis" / f"epoch_{epoch}"
                    for b_i in range(min(images.size(0), save_vis_max_samples - vis_saved_val)):
                        save_line_vis(
                            images[b_i], masks[b_i], logits[b_i],
                            vis_dir / f"val_{vis_saved_val}.png",
                        )
                        vis_saved_val += 1

        val_loss = val_running / max(len(val_dl), 1)

        scheduler.step()
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"[line] E{epoch} train={train_loss:.4f} val={val_loss:.4f} lr={lr:.2e} t={elapsed:.1f}s")

        csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.6e}", f"{elapsed:.1f}"])
        metrics_fp.flush()

        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss),
            "config": cfg_dict,
        }
        torch.save(ckpt_state, CHECKPOINT_DIR / "last.pt")
        torch.save(ckpt_state, experiment_ckpt_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_state, CHECKPOINT_DIR / "best.pt")
            torch.save(ckpt_state, experiment_ckpt_dir / "best.pt")
            print(f"  ★ New best val_loss: {best_val_loss:.4f}")

    metrics_fp.close()
    return {"best_val_loss": best_val_loss}
