#!/usr/bin/env python
"""List / backfill / prune training checkpoints under ``outputs/`` (issue #533).

Checkpoints dominate ``outputs/`` storage (~1.4 GB each for PLCS). The #533
workflow replaces them with test-split prediction arrays: once a checkpoint's
predictions are saved, the checkpoint can be deleted and any new metric is still
recomputable from ``pred_test.npz``.

This helper is **manual** (issue: "ckpt 消去はマニュアル") and conservative:

- default (no flags): dry-run listing of every ``*.ckpt`` with sizes + total.
- ``--backfill``: for each checkpoint, rebuild the model **from the config saved
  inside the checkpoint** (``hyper_parameters``), run test inference, and write
  ``<version_dir>/predictions/pred_test.npz``. Checkpoints that fail to load
  cleanly (architecture drift, missing config) are skipped and reported.
- ``--delete``: remove a checkpoint **only if** a verified prediction npz exists
  next to it. Combine with ``--backfill`` to backfill-then-delete in one pass.

Usage:
    .venv/bin/python .agents/skills/training-queue/scripts/prune_ckpts.py            # list
    .venv/bin/python .../prune_ckpts.py --backfill --device cuda                     # backfill
    .venv/bin/python .../prune_ckpts.py --backfill --delete --device cuda            # + prune
    .venv/bin/python .../prune_ckpts.py --delete                                     # prune already-backfilled
    .venv/bin/python .../prune_ckpts.py --glob 'outputs/plcs/**/*.ckpt'              # scope
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return here.parents[4]


def human(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            return f"{val:.1f}{unit}"
        val /= 1024
    return f"{val:.1f}TB"


def detect_task(ckpt: Path) -> str | None:
    parts = {p.lower() for p in ckpt.parts}
    if "plcs" in parts:
        return "plcs"
    if "blcs" in parts:
        return "blcs"
    return None


def version_dir(ckpt: Path) -> Path:
    """``.../logs/version_N/checkpoints/x.ckpt`` -> ``.../logs/version_N``."""
    return ckpt.parent.parent


def predictions_npz(ckpt: Path) -> Path:
    return version_dir(ckpt) / "predictions" / "pred_test.npz"


def verify_npz(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            return "scene_ids" in data and len(data["scene_ids"]) > 0
    except Exception:
        return False


def build_runner(task: str):
    if task == "plcs":
        from src.tasks.plcs.training.runner import PLCSTrainingRunner

        return PLCSTrainingRunner()
    from src.tasks.blcs.training.runner import BLCSTrainingRunner

    return BLCSTrainingRunner()


def build_module_cls(task: str):
    if task == "plcs":
        from src.tasks.plcs.training.lightning_module import PLCSLightningModule

        return PLCSLightningModule
    from src.tasks.blcs.training.lightning_module import BLCSLightningModule

    return BLCSLightningModule


def backfill_one(ckpt: Path, task: str, device: str) -> tuple[bool, str]:
    """Rebuild the model from the checkpoint and save test predictions.

    Returns (ok, message). Architecture drift / config problems return (False, ...).
    """
    import pytorch_lightning as pl
    import torch

    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"load failed: {exc}"
    config = (blob.get("hyper_parameters") or {}).get("config")
    if config is None:
        return False, "no config in checkpoint hyper_parameters"

    runner = build_runner(task)
    module_cls = build_module_cls(task)
    try:
        runner.prepare_config(config)
        datamodule = runner.build_datamodule(config)
        module = module_cls.load_from_checkpoint(str(ckpt), config=config)
    except Exception as exc:  # noqa: BLE001
        return False, f"reconstruct failed: {exc}"

    # Predictions land next to the checkpoint's version dir.
    os.environ["TENNIS_REPRO_DIR"] = str(version_dir(ckpt))
    accelerator = "gpu" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    try:
        trainer.test(module, datamodule=datamodule)
    except Exception as exc:  # noqa: BLE001
        return False, f"test failed: {exc}"
    finally:
        os.environ.pop("TENNIS_REPRO_DIR", None)

    npz = predictions_npz(ckpt)
    if verify_npz(npz):
        return True, f"saved {npz.relative_to(repo_root())}"
    return False, "test ran but npz missing/empty"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="outputs/**/*.ckpt", help="ckpt glob (repo-root relative)")
    p.add_argument("--backfill", action="store_true", help="save test predictions from each ckpt")
    p.add_argument("--delete", action="store_true", help="delete ckpt when verified npz exists")
    p.add_argument("--device", default="cuda", help="cuda|cpu for backfill")
    p.add_argument("--limit", type=int, default=0, help="process at most N ckpts (0 = all)")
    args = p.parse_args()

    root = repo_root()
    os.chdir(root)
    ckpts = sorted(root.glob(args.glob))
    if not ckpts:
        print(f"no checkpoints match {args.glob}")
        return 0
    if args.limit:
        ckpts = ckpts[: args.limit]

    total = sum(c.stat().st_size for c in ckpts)
    print(f"{len(ckpts)} checkpoint(s), {human(total)} total under {args.glob}\n")

    if not (args.backfill or args.delete):
        for c in ckpts:
            npz = predictions_npz(c)
            flag = "[npz✓]" if verify_npz(npz) else "[no-npz]"
            print(f"  {human(c.stat().st_size):>9}  {flag:9}  {c.relative_to(root)}")
        print("\n(dry-run) re-run with --backfill and/or --delete to act.")
        return 0

    freed = 0
    done = skipped = deleted = 0
    for c in ckpts:
        task = detect_task(c)
        rel = c.relative_to(root)
        if task is None:
            print(f"SKIP  {rel}  (unknown task; not plcs/blcs)")
            skipped += 1
            continue
        if args.backfill and not verify_npz(predictions_npz(c)):
            ok, msg = backfill_one(c, task, args.device)
            print(f"{'OK  ' if ok else 'SKIP'}  {rel}  ({msg})")
            if ok:
                done += 1
            else:
                skipped += 1
                continue
        if args.delete:
            if verify_npz(predictions_npz(c)):
                size = c.stat().st_size
                c.unlink()
                freed += size
                deleted += 1
                print(f"DEL   {rel}  (freed {human(size)})")
            else:
                print(f"KEEP  {rel}  (no verified npz; not deleting)")

    print(
        f"\nbackfilled={done} deleted={deleted} skipped={skipped} freed={human(freed)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
