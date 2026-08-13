---
id: group-i719-plcs-reference-ablation
type: group
title: PLCS CourtKP14/KP7 reference ablation (#719)
issue: 719
members:
- run-i719-plcs-kp14
- run-i719-plcs-kp7-no-reference
- run-i719-plcs-kp7-reference
parents: []
tags:
- plcs
- tracking
- court-kp7
- reference-ablation
---

## まとめ

seed 719、固定 32/8/8 scene split、T=64、batch 8、3 epoch の同一予算で、PLCS の ordered KP14、unordered KP7 no-reference、unordered KP7 reference-conditioned を比較した。

| 条件 | total position error | Y MAE | Y sign accuracy | heading error | lifecycle F1 |
|---|---:|---:|---:|---:|---:|
| KP14 baseline | 1.134778 | 7.104686 m | 0.375000 | 45.087929° | 0.380952 |
| KP7 no-reference | 0.944179 | **3.868129 m** | **0.750000** | **29.407085°** | 0.344828 |
| KP7 reference | **0.811520** | 6.264746 m | 0.375000 | 98.846252° | **0.388889** |

reference-conditioned KP7はtotal position errorを改善したが、no-referenceに対してY MAEが2.396617 m、heading errorが69.439167°悪化し、Y sign accuracyも0.375000低下した。paired Y consistencyも6.312500 mでno-referenceの3.796875 mより悪い。したがってproduction採用条件を満たさず、defaultはKP14のまま維持する。actual reference fusionはC=7/N=4/D=4、B=1/V=3/T=64、10 warmups/50 repeatsで3.216010 ms、43.975586 MBだった。

no-referenceの高いY signは固定splitのgeometry/side分布shortcutの可能性がある。次はview orderとreference sideを均衡化した複数seed counterfactual testとheading warmupを行い、Y MAE・Y sign・heading consistencyの全てで改善するかを検証する。
