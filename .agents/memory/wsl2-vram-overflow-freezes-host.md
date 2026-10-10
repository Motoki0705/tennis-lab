---
title: "WSL2 で 16GB の VRAM を使い切ると、エラーにならずホスト全体が周期的に固まる"
type: environment
applies_to: all
source: "#540 PLCS multiview 学習（2026-06-20）"
created: 2026-06-20
last_verified: 2026-10-10
evidence: "GPU は RTX 5060 Ti 16311 MiB（nvidia-smi）。src/tasks/plcs/configs/data/_multiview.yaml は batch_size 4。フリーズ自体は2026-06-20の観察で、再現はしていない"
---

WSL2 の NVIDIA ドライバは、VRAM があふれると OOM にせず、GPUメモリをシステムRAMへ退避する（sysmem fallback）。そのとき、学習の各iterationでWindowsホスト全体が約1秒固まる。目印は、GPUメモリが約97%に張り付き、利用率が1%と95%の間を往復し、Linux側のload averageは低いままであること。

このため `src/tasks/plcs/configs/data/_multiview.yaml` の `batch_size` は 4 になっている（8 だと約16.3GBの上限に達した）。

**使い方:** 大きなモデルや長い系列では、ピークVRAMに数GBの余裕を残す。batch_size を上げたら、学習開始直後に `nvidia-smi` で使用量を確かめる。GPUを使う作業（backfill など）を、training queue の学習と同時に走らせない。
