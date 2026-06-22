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
    .venv/bin/python .../prune_ckpts.py --repro-dir .training_queue/repro/<id> --delete  # one run (queue hook)

Single-run mode (``--repro-dir``) gates on the run's own repro bundle instead of
the global glob: it deletes only the checkpoints pointed at by
``<repro>/output_dir.txt`` and only when ``<repro>/predictions/pred_test.npz``
verifies. The training queue's optional ``--prune-ckpt`` flag calls this after a
successful run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


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


def prune_from_repro_dir(repro_dir: Path, do_delete: bool) -> int:
    """Delete one run's checkpoints, gated on its repro bundle (issue #533).

    This powers the training queue's optional post-run auto-prune. A queued run's
    gitignored ``.training_queue/repro/<jobid>/`` holds both the reproducibility
    bundle and the test-split predictions (``predictions/pred_test.npz``); the
    runner writes ``output_dir.txt`` there pointing at the run's checkpoint dir.
    A checkpoint is removed **only if** a verified ``pred_test.npz`` exists in
    this repro dir, so every metric stays recomputable; the npz itself is never
    touched. Conservative and self-contained: any missing precondition keeps the
    checkpoints and returns 0, so a queue worker never treats a successful run as
    failed.
    """
    npz = repro_dir / "predictions" / "pred_test.npz"
    if not verify_npz(npz):
        print(f"KEEP  {repro_dir}  (no verified pred_test.npz; not pruning)")
        return 0
    pointer = repro_dir / "output_dir.txt"
    if not pointer.exists():
        print(f"KEEP  {repro_dir}  (no output_dir.txt pointer; cannot locate ckpts)")
        return 0
    try:
        pointer_value = pointer.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"KEEP  {repro_dir}  (cannot read output_dir.txt: {exc})")
        return 0
    if not pointer_value:
        print(f"KEEP  {repro_dir}  (empty output_dir.txt pointer)")
        return 0
    ckpt_dir = Path(pointer_value)
    if not ckpt_dir.is_absolute():
        ckpt_dir = repo_root() / ckpt_dir
    ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    if not ckpts:
        print(f"KEEP  {repro_dir}  (no *.ckpt under {ckpt_dir}; already pruned?)")
        return 0
    total = sum(c.stat().st_size for c in ckpts)
    if not do_delete:
        print(
            f"(dry-run) would delete {len(ckpts)} ckpt(s), {human(total)} under {ckpt_dir}"
        )
        for c in ckpts:
            print(f"  {human(c.stat().st_size):>9}  {c}")
        print("  re-run with --delete to remove (pred_test.npz is verified).")
        return 0
    freed = 0
    deleted = 0
    for c in ckpts:
        size = c.stat().st_size
        try:
            c.unlink()
        except OSError as exc:  # noqa: PERF203
            print(f"KEEP  {c}  (unlink failed: {exc})")
            continue
        freed += size
        deleted += 1
        print(f"DEL   {c}  (freed {human(size)})")
    print(
        f"pruned {deleted}/{len(ckpts)} ckpt(s) for {repro_dir.name}; "
        f"freed {human(freed)} "
        f"(pred_test.npz retained at {npz})"
    )
    return 0


def build_runner(task: str) -> Any:
    if task == "plcs":
        from src.tasks.plcs.training.runner import PLCSTrainingRunner

        return PLCSTrainingRunner()
    from src.tasks.blcs.training.runner import BLCSTrainingRunner

    return BLCSTrainingRunner()


def build_module_cls(task: str) -> Any:
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

    # Local training checkpoints are trusted and embed an omegaconf config, which
    # PyTorch 2.6's default weights_only=True refuses to unpickle. Match the
    # runner's resume path (BaseTrainingRunner.resume_checkpoint_load_env).
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
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
    accelerator = (
        "gpu" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
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
    p.add_argument(
        "--glob", default="outputs/**/*.ckpt", help="ckpt glob (repo-root relative)"
    )
    p.add_argument(
        "--backfill", action="store_true", help="save test predictions from each ckpt"
    )
    p.add_argument(
        "--delete", action="store_true", help="delete ckpt when verified npz exists"
    )
    p.add_argument("--device", default="cuda", help="cuda|cpu for backfill")
    p.add_argument(
        "--limit", type=int, default=0, help="process at most N ckpts (0 = all)"
    )
    p.add_argument(
        "--repro-dir",
        type=Path,
        default=None,
        help="prune ONE run's ckpts via its .training_queue/repro/<jobid> bundle "
        "(verified pred_test.npz + output_dir.txt pointer); ignores --glob",
    )
    args = p.parse_args()

    # Single-run mode (training-queue post-run auto-prune): scoped + self-gating.
    if args.repro_dir is not None:
        repro = (
            args.repro_dir
            if args.repro_dir.is_absolute()
            else (Path.cwd() / args.repro_dir)
        ).resolve()
        os.chdir(repo_root())
        return prune_from_repro_dir(repro, args.delete)

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
