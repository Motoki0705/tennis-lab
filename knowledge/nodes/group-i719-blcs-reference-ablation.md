---
id: group-i719-blcs-reference-ablation
type: group
title: BLCS CourtKP14/KP7 reference ablation (#719)
issue: 719
members:
- run-i719-blcs-kp14
- run-i719-blcs-kp7-no-reference
- run-i719-blcs-kp7-reference
parents: []
tags:
- blcs
- tracking
- court-kp7
- reference-ablation
---

## まとめ

seed 719、固定 32/8/8 scene split、T=64、batch 8、3 epoch の同一予算で、ordered KP14、unordered KP7 no-reference、unordered KP7 reference-conditioned を比較した。

| 条件 | total position error | Y MAE | Y sign accuracy | lifecycle F1 |
|---|---:|---:|---:|---:|
| KP14 baseline | 2.295817 | **5.893940 m** | **0.669922** | 0.543906 |
| KP7 no-reference | 2.361911 | 6.190108 m | 0.638672 | 0.441627 |
| KP7 reference | **2.290416** | 7.348188 m | 0.433594 | **0.836601** |

reference-conditioned KP7はno-referenceよりY MAEが1.158080 m悪化し、Y sign accuracyも0.205078低下した。lifecycle F1とtotal position errorは良いが、Issue #719が定めるY軸一意化のproduction採用条件を満たさないため、defaultはKP14のまま維持する。paired Y consistencyも4.093750 mでno-referenceの1.265625 mより悪い。actual reference fusionはC=7/N=4/D=4、B=1/V=3/T=64、10 warmups/50 repeatsで4.065558 ms、43.828125 MBだった。

観測から言えるのは単一seed・3 epochの比較までである。次は複数seed・長期学習とreference-side-balanced counterfactual testを実施し、KP7 referenceがno-referenceをY MAEとY signの双方で上回ることを採用条件とする。
