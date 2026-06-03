"""Overview:
Pull a workflow `.tar.zst` archive from a Drive path or rclone remote to local storage.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/drive_pull_archive.py drive_archive=/content/drive/MyDrive/tennis_lab/data/dino_workflow/pseudo_raw_round001.tar.zst local_archive=data/dino_workflow/archives/pseudo_raw_round001.tar.zst
    .venv/bin/python experiments/dino_lora_workflow/scripts/drive_pull_archive.py drive_archive=google:tennis_lab/data/dino_workflow/pseudo_raw_round001.tar.zst local_archive=data/dino_workflow/archives/pseudo_raw_round001.tar.zst

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/drive_pull_archive.yaml`.
    - Plain filesystem paths use `shutil.copy2`; `name:path` locations use `rclone copyto`.
    - A sidecar `<archive>.transfer.json` is written next to the copied local archive by default.
"""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from drive_push_archive import transfer_archive


def drive_pull_archive(cfg: DictConfig) -> dict[str, object]:
    return transfer_archive(cfg, direction="pull")


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="drive_pull_archive",
)
def main(cfg: DictConfig) -> None:
    summary = drive_pull_archive(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
