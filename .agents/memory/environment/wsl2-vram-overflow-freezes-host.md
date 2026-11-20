---
title: "WSL2のPLCS学習では16GB GPUのVRAM上限付近でホストが周期的に固まった"
summary: "WSL2のPLCS multiview学習では16GB GPUのVRAM上限付近でWindowsホストが短周期に固まった。ピークVRAMに余裕を残す。"
applies_to: all
source: "#540 PLCS multiview 学習（2026-06-20）"
created: 2026-06-20
last_verified: 2026-10-10
evidence: "GPU は RTX 5060 Ti 16311 MiB（nvidia-smi）。src/tasks/plcs/configs/data/_multiview.yaml は batch_size 4。フリーズ自体は2026-06-20の観察で、再現はしていない"
citations:
  - path: "src/tasks/plcs/configs/data/_multiview.yaml"
    symbol: "batch_size: 4"
    note: "観察後の対象設定"
---

2026-06-20のPLCS multiview学習では、GPUメモリが約97%に張り付いて、学習の各iterationでWindowsホスト全体が約1秒固まった。利用率は1%と95%の間を往復し、Linux側のload averageは低いままだった。GPUメモリがシステムRAMへ退避された可能性がある。

このため `src/tasks/plcs/configs/data/_multiview.yaml` の `batch_size` は 4 になっている（8 だと約16.3GBの上限に達した）。

**使い方:** 大きなモデルや長い系列では、ピークVRAMに数GBの余裕を残す。batch_size を上げたら学習プロセス内のCUDA allocator統計で使用量を確かめる。GPUの作業は [training queue](../../skills/training-queue/SKILL.md) の資源予約に従う。
