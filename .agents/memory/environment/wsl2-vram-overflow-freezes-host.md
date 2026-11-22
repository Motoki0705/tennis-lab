---
title: WSL2のPLCS学習では16GB GPUのVRAM上限付近でホストが周期的に固まった
summary: WSL2のPLCS multiview学習では16GB GPUのVRAM上限付近でWindowsホストが短周期に固まった。ピークVRAMに余裕を残す。
applies_to: all
source: '#540 PLCS multiview 学習（2026-06-20）'
created: 2026-06-20
last_verified: 2026-10-10
evidence: 2026-06-20のフリーズは当時の観察。_multiview.yamlのbatch_sizeは4。#535のdeep16再計測ノードにはVRAM制約でbatch_size 4を使用した記録がある。
citations:
- path: src/tasks/plcs/configs/data/_multiview.yaml
  symbol: 'batch_size: 4'
  note: 共通設定の値
- path: knowledge/nodes/plcs/000043-run-i535-asym-deep16-rerun.md
  symbol: VRAM 制約で
  note: 別の再計測でバッチを下げた根拠
---

2026-06-20のPLCS multiview学習では、RTX 5060 Ti（16,311 MiB）のVRAM使用量が約97%のとき、Windowsホストがiterationごとに約1秒固まった。GPU利用率は1%と95%の間を往復し、Linux側のload averageは低いままだった。batch_size 8で約16.3GBに達した記録がある。ホストRAMへの退避は当時の仮説である。

共通設定ではbatch_sizeは4。大きいモデルや長い系列に変えた場合は、学習プロセス内のCUDA allocator統計でピークを確かめ、VRAMに余裕を持たせる。
