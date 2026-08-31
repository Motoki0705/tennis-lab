---
id: group-i801-beta005-presence-competition-ablation
type: group
title: beta005 presence competition アブレーション（#801）
issue: 801
members:
- run-i801-dref-pose-beta005-presence-comp-hneg050-s42
- run-i801-eval-beta005-presence-comp-hneg050-e00-thr050-currentmetrics
- run-i801-eval-beta005-presence-head-hneg050-e01-thr050-compcontrol
- run-i801-dref-pose-beta005-presence-comp-centered-hneg050-s42
- run-i801-eval-beta005-presence-comp-centered-e02-thr050-currentmetrics
- run-i801-dref-pose-beta005-presence-comp-centered-pair010m05-s42
- run-i801-eval-beta005-presence-comp-centered-pair010m05-e00-thr050-currentmetrics
parents: []
tags: [plcs, tracking, pose, presence, competition, deepsets, zero-mean, pairwise, ablation, current-metrics, fair-contract, beta005]
---

## まとめ

hard-negative `0.5` epoch 1を固定対照として、post-hoc DeepSets presence competitionを3構成で比較した。uncentered / controlはcommit `1275bdb1`、centered 2構成はcommit `a4279f40`で評価したが、この間にcurrent tracking metric集計の変更はない。全評価はpresence threshold `0.5`、duplicate / ID switch距離各`0.05 m`、同一test split、T=128、V=6、reference camera `camera_2`を共有する。control再評価bundleは既存current-metrics nodeとpredictions / metrics / diagnosticのbyte hashまで一致した。

| checkpoint | precision | recall | F1 | duplicate | missed | exact count | GT1 all4 | GT2 all4 | stable zero-margin violation | margin 0.5 hinge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard-negative control epoch 1 | **0.510584** | 0.981537 | **0.671911** | **44.35** | 4.00 | **0.231016** | **0.768659** | **0.996453** | **0.700344** | 0.502529 |
| uncentered DeepSets epoch 0 | 0.507579 | 0.983064 | 0.670431 | 46.23 | **3.52** | 0.229219 | 0.830895 | 0.998226 | 0.728488 | 0.486175 |
| zero-mean DeepSets epoch 2 | 0.506005 | **0.983249** | 0.669057 | 47.67 | 3.54 | 0.227734 | 0.861394 | 0.998226 | 0.714416 | 0.480906 |
| zero-mean + pairwise epoch 0 | 0.507579 | **0.983249** | 0.670431 | 47.35 | 3.57 | 0.230547 | 0.848500 | **0.996453** | 0.727678 | **0.478444** |

結論は3候補すべて`REJECT`である。controlのF1 `0.671911`、duplicate `44.35`、exact-count `0.231016`、GT 1人all4 `0.768659`を同時に超えるDeepSets候補はなく、recall / missedの小改善は余剰query発火を増やした代償だった。全候補でGT 2人all4率が`99.65%`以上、exact-count率が`23.06%`以下で、query過活性化は未解決である。

uncentered branchはcontrolとの差分residual二乗エネルギーの`48.22%`がframe共通成分で、query間residual相関平均も`0.723`だった。scene内で共通shiftが時間方向に概ね一定であり、人数競合ではなくscene-level common gateとして使われた。zero-mean化により共通成分比は`0.015%`以下へ落ちたが、best checkpointのstable zero-margin違反はcontrol `0.700344`に対し`0.714416`、exact-countは`0.227734`で、ranking / countはなお悪化した。したがってcommon gate除去だけでは根因を解けない。

pairwise追加はmargin `0.5` hinge平均をcontrol `0.502529`から`0.478444`へ縮めた一方、hardest-pairのzero-margin違反は`0.727678`、exact-countは`0.230547`、GT 1人all4は`0.848500`でcontrolより悪かった。つまり学習目的に近いhinge幅だけが改善し、符号・threshold後のactive数へ変換されていない。weight / marginの追加sweepは行わない。

full-run最終stateも同じ結論である。

| full final state | F1 | duplicate | exact count | GT1 all4 | stable zero-margin violation | margin 0.5 hinge |
|---|---:|---:|---:|---:|---:|---:|
| uncentered + hard-negative | 0.667433 | 48.16 | 0.224063 | 0.888421 | 0.731525 | 0.479152 |
| zero-mean + hard-negative | 0.667860 | **48.01** | **0.226094** | **0.880734** | 0.717757 | 0.478463 |
| zero-mean + pairwise | **0.667860** | 48.05 | 0.225781 | 0.883213 | **0.714720** | **0.477536** |

3系統のepoch 0 / best / last checkpointをsourceと比較すると、追加competition branch以外の共通175 state tensorは全てbitwise同一だった。eval bundleでもposition `4.929797 m`、angular `33.645799°`、canonical MPJPE `0.174605 m`、reprojection `155.018780 px`が4候補で一致し、pose保持は成立している。したがって今回の不採用はpose破壊ではなくpresence competition自体のtracking効果不足による。

次はhard-negative controlへ戻し、post-hoc DeepSets residualの探索を止める。query interaction本体で人数競合を表現する、予測人数に応じたtop-kを構造化する、またはHungarian-unmatched queryのlogit符号を直接制約する施策を検討する。次候補も同じmetric contractでF1 / duplicate / missed / exact-count / GT人数別all4 / stable ranking / pose bitwise保持を同時に判定する。
