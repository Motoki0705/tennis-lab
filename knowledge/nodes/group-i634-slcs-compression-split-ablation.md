---
id: group-i634-slcs-compression-split-ablation
type: group
title: SLCS DINO圧縮・trunk分離アブレーション (#634)
issue: 634
members:
- run-i634-slcs-overfit-no-dino
- run-i634-slcs-overfit-dino
- run-i634-slcs-overfit-dino-down2-shared
- run-i634-slcs-overfit-split-no-dino
- run-i634-slcs-overfit-split-dino
- run-i634-slcs-overfit-split-dino-down2
parents: []
tags: [slcs, overfit, dino, patch-compression, split-trunk]
---

## まとめ

単一clip過学習における共有/完全分離trunkと、DINO patch 448/112 tokenの2軸比較。

| 構成 | player位置 | yaw | ball位置 |
|---|---:|---:|---:|
| 共有・DINOなし | 0.484m | 9.849° | 2.105m |
| 共有・DINO 448 | 0.470m | 7.762° | 1.954m |
| 共有・DINO 112 | **0.467m** | 8.547° | 1.745m |
| 分離・DINOなし | 0.543m | 8.319° | 1.827m |
| 分離・DINO 448 | 0.512m | **5.751°** | **1.616m** |
| 分離・DINO 112 | 0.532m | 6.452° | 1.698m |

完全分離はDINO有無の両方でyawとballを改善したが、player位置を悪化させた。したがってrotation勾配の負の転移を遮断する効果と、共有表現によるplayer位置への正の転移が同時に存在する可能性がある。次はshared 1層 + task 1層の中間配分が優先候補である。

DINO 2×2圧縮は共有trunkではplayer位置を維持しballを改善したが、完全分離では非圧縮に劣った。task固有branchほど局所visual情報を利用できるため、圧縮との相互作用が構成依存になった可能性がある。すべて同一train/val/test windowであり、結論はmemorization能力に限定する。
