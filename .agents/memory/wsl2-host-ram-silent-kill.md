---
title: "WSL2 のホストRAM不足では、学習プロセスがtracebackなしで黙って殺される"
type: environment
applies_to: all
source: "DINOv3 LoRA-SSL 学習（2026-06-30）、court 学習中のWSL2再起動（2026-07-05）"
created: 2026-06-30
last_verified: 2026-06-30
evidence: "再現はしていない。DataLoader の num_workers を減らすと停止位置が後ろへずれたという2026-06-30の観察だけが根拠"
---

画像約30万枚の DINOv3 SSL 学習が、ログの途中で何も出さずに止まった（3回）。VRAMは約4.8GBで余裕があった。停止位置は `num_workers` に比例して前後し、新しいプロセスで再開するとリセットされた。DataLoader worker の RSS が少しずつ増え（大きなパスのタプルの copy-on-write）、WSL2のホストRAMの上限に達したと考えている。WSL2そのものが再起動することもある（2026-07-05）。

**使い方:** tracebackなしで止まったら、まずホストRAMを疑う。`num_workers` を減らす、checkpoint の間隔を詰める、自動で再開する watchdog を付ける、GPU学習中に重いCPUジョブを並行させない、を試す。
