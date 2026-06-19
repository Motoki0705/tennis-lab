---
id: group-i519-mcmc
type: group
title: MCMC/SGLD 学習戦略 (#519)
issue: 519
members:
- run-i519-mcmc-ns01
- run-i519-mcmc-ns03
parents: []
tags:
- plcs
- rotation
- mcmc
---

## まとめ

「MCMC（SGLD: 重みへの Langevin ノイズ注入）を学習戦略として 180° 反転の局所最適を脱出できるか」を
検証したアブレーション（`canonical` ベースライン + 全パラ SGLD, noise_scale 振り）。

- `ns=0.1`: `73.58° / 0.512m`、`ns=0.3`: `76.11° / 0.754m`。いずれもノイズなしベースライン
  （約 `61.6° / 0.260m`）より**悪化**し、noise_scale を上げるほど単調に劣化。
- 原因: AdamW は勾配を正規化して約 lr のステップを踏むため、SGLD ノイズ std `sqrt(2*lr)≈0.014` が
  信号を上書きしてしまう。

**結論はネガティブ**: MCMC は 180° 反転脱出に効かない。正解は損失ベースの **#518**（wrapped-angle
損失で反転サドルの勾配を回復＋分離 trunk）。MCMC 機構自体は任意オプション (`mcmc.yaml`) として残置。
