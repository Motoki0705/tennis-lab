---
id: group-i530-canonical-head-ablation
type: group
title: canonical pose head 構造化アブレーション (#530)
issue: 530
members:
- run-i530-direct-baseline
- run-i530-mean-residual
- run-i530-parent-delta
- run-i530-bone-direction
parents: []
tags:
- plcs
- canonical
- structured-head
- head-ablation
- split-trunk
---

## 考察 / Findings

canonical pose の出力表現を direct、平均ポーズ残差、親相対デルタ、骨方向の4案で比較した。

| head | yaw誤差 | 位置誤差 | canonical MPJPE | 骨長相対誤差 |
|---|---:|---:|---:|---:|
| direct | **9.64°** | **0.247 m** | 0.1384 m | 4.29% |
| mean residual | 13.43° | 0.298 m | 0.1362 m | 4.19% |
| parent delta | 13.32° | 0.285 m | 0.1367 m | 4.61% |
| bone direction | 11.01° | 0.252 m | **0.1342 m** | **4.12%** |

静的な関節角・torsion 指標は全案でほぼ同等だった。構造化により canonical geometry
は改善できるが、hard replacement は yaw・位置推定を損なう。次は direct を主 head
として維持し、bone-direction を重み `0.1–0.3` の補助 head にする案を優先する。
次点は bone decode に bounded joint residual を加える案、単一 scale を
sequence ごとの12骨長 residual に置き換える案。
