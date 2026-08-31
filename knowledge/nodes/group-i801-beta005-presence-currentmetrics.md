---
id: group-i801-beta005-presence-currentmetrics
type: group
title: beta005 presence公平再評価（current metrics、#801）
issue: 801
members:
- run-i801-dref-pose-beta005-presence-head-pair010m05-s42
- run-i801-eval-beta005-presence-head-pair010m05-e00-thr050-currentmetrics
- run-i801-eval-beta005-presence-head-inact1-e03-thr050-currentmetrics
- run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
- run-i801-eval-beta005-e69-thr050-currentmetrics
parents: []
tags: [plcs, tracking, pose, presence, evaluation, current-metrics, fair-contract, beta005]
---

## まとめ

presence-head候補4 checkpointを、commit `fafff3ae`のcurrent metric aggregation、presence threshold `0.5`、duplicate / ID switch距離各`0.05 m`、同一test splitで公平比較した。全候補のID switchは`0.13`である。

| checkpoint | precision | recall | F1 | duplicate | missed | inactive FP | GT1 all4 | GT2 all4 | exact count | stable zero-margin violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pairwise 0.1 epoch 0 | 0.505607 | 0.985120 | 0.668627 | 46.77 | 3.14 | 172.12 | 84.60% | 99.94% | 22.53% | 70.16% |
| inactive 1.0 epoch 3 | 0.510172 | 0.982668 | **0.672165** | 44.75 | 3.77 | 168.48 | 78.45% | 99.65% | 23.02% | 70.91% |
| hard-negative 0.5 epoch 1 | **0.510584** | 0.981537 | 0.671911 | **44.35** | 4.00 | **167.96** | **76.87%** | 99.65% | **23.10%** | **70.03%** |
| source epoch 69 | 0.502481 | **0.991527** | 0.667258 | 48.32 | **1.90** | 175.28 | 90.95% | 99.94% | 22.02% | 70.52% |

headline bestはinactive `1.0` epoch 3、過活性抑制とのbalanced bestはhard-negative `0.5` epoch 1である。hard-negativeは最高precision、最少duplicate / inactive FP、最低GT 1人all4率、最高exact-count率を得た代わりに、inactive `1.0`よりF1が`0.000254`低く、missedが`0.23`多い。

pairwise weight `0.1` / margin `0.5`は不採用とする。full run最終stateはF1`0.667681`、GT 1人all4率`92.59%`、exact-count率`21.86%`、stable margin 0.5違反率`85.93%`であった。best epoch 0でもF1`0.668627`とGT 1人all4率`84.60%`はinactive / hard-negativeに及ばず、aligned stable rankingもhardest gap`-0.233844`、zero-margin違反`70.16%`で実質的な改善がない。

全候補でGT 2人時の4-query全発火率が`99.65%`以上であり、全体exact-count率も最高`23.10%`に過ぎない。このaccuracyはGT 0人・4人frameで押し上げられており、GT 1人・2人のexact countはほぼ失敗している。したがってquery過活性化は未解決で、F1の小差だけをもって完了とは扱わない。

次はhard-negative epoch 1またはinactive `1.0` epoch 3を起点に、Hungarian assignmentでunmatchedのqueryを直接抑える目的、またはquery間競合を明示する仕組みを試す。checkpoint選択ではGT 1人・2人all4率とexact-count率をprimary gateとし、F1、recall、missed、pose不変も同時に守る。
