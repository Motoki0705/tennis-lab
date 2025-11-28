"""
How to use:
model = ...

callbacks = build_default_callbacks(
    monitor="val/loss",
    mode="min",
    patience=20,
    save_top_k=5,
    enable_batch_size_finder=False,
    enable_backbone_finetuning=True,
)

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=callbacks,
    # その他引数...
)

trainer.fit(model)
"""
from __future__ import annotations

from typing import List

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from lightning.pytorch.callbacks.batch_size_finder import BatchSizeFinder
from lightning.pytorch.callbacks.finetuning import BackboneFinetuning


def build_default_callbacks(
    # 監視系共通
    monitor: str = "val/loss",
    mode: str = "min",

    # EarlyStopping
    patience: int = 10,
    min_delta: float = 0.0,

    # ModelCheckpoint
    save_top_k: int = 3,
    ckpt_dir: str = "checkpoints",
    ckpt_filename: str = "{epoch:02d}-{step}",

    # LR monitor
    lr_logging_interval: str = "epoch",  # or "step"

    # TQDM
    tqdm_refresh_rate: int = 10,

    # BatchSizeFinder
    enable_batch_size_finder: bool = False,
    bsf_mode: str = "power",           # or "binsearch"
    bsf_batch_arg_name: str = "batch_size",

    # BackboneFinetuning
    enable_backbone_finetuning: bool = False,
    unfreeze_at_epoch: int = 5,
    backbone_initial_lr: float | None = None,
) -> List[Callback]:
    """Trainer に渡すデフォルトコールバック一式を生成する。"""
    callbacks: List[Callback] = []

    # ---- ModelCheckpoint ----
    callbacks.append(
        ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=True,
            dirpath=ckpt_dir,
            filename=ckpt_filename,
            auto_insert_metric_name=False,
        )
    )

    # ---- EarlyStopping ----
    callbacks.append(
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            verbose=True,
        )
    )

    # ---- LearningRateMonitor ----
    callbacks.append(
        LearningRateMonitor(logging_interval=lr_logging_interval)
    )

    # ---- TQDMProgressBar ----
    callbacks.append(
        TQDMProgressBar(refresh_rate=tqdm_refresh_rate)
    )

    # ---- BatchSizeFinder (optional) ----
    if enable_batch_size_finder:
        callbacks.append(
            BatchSizeFinder(
                mode=bsf_mode,
                batch_arg_name=bsf_batch_arg_name,
            )
        )

    # ---- BackboneFinetuning (optional) ----
    if enable_backbone_finetuning:
        callbacks.append(
            BackboneFinetuning(
                unfreeze_backbone_at_epoch=unfreeze_at_epoch,
                backbone_initial_lr=backbone_initial_lr,
                # 必要なら lambda_func や backboneスクジューラもここに足す
            )
        )

    return callbacks
