---
id: group-i832-blcs-observation-tracking
type: group
title: BLCS observation-based deterministic 2D tracking comparison (#832)
issue: 832
members:
- run-i832-blcs-legacy-slot-baseline
- run-i832-blcs-tracker-conservative
- run-i832-blcs-tracker-permissive
tags:
- blcs
- deterministic-2d-tracking
- tracking
- issue-832
---

## まとめ

### 比較条件と headline metrics

3 run は seed 832、`blcs_track_query`、100 epoch、同一 `blcs/multi_object` split、FP augmentation
無効で揃えた。baseline は GT lifecycle の random slot、候補 2 run は noisy 2D observation の
deterministic association を使う。

| run | association | position error (m) | presence F1 | ID switches | precision / recall | duplicate active tracks | missed GT frames |
|---|---|---:|---:|---:|---:|---:|---:|
| [[run-i832-blcs-legacy-slot-baseline]] | legacy random slot | 5.4342134 | 0.8786702 | 0.64 | 0.8085196 / 0.9609743 | 80.22 | 26.76 |
| [[run-i832-blcs-tracker-conservative]] | `0.04`, 2 frames | **5.4308771** | **0.8811509** | 0.66 | **0.8114234 / 0.9638893** | 86.19 | **24.63** |
| [[run-i832-blcs-tracker-permissive]] | `0.10`, 8 frames | 5.4355659 | 0.8787612 | **0.60** | 0.8079256 / 0.9629848 | 83.53 | 25.32 |

数値は各 run の `metrics.json` / `diagnostic_metrics.json` の test split 実測値である。position、F1、
precision、recall、missed GT frames は conservative が 3 候補中最良だった。permissive は ID
switches が最良だが、precision と duplicate/error 側を含む総合観点では conservative を上回らない。

### 収束と failure evidence

`curves.png` は各 run について `kg_curves.py` で生成した。train loss の初期→最終は baseline
`0.6874→0.1834`、conservative `0.6873→0.1827`、permissive `0.6872→0.1823`。val loss の最終値は
それぞれ `0.2002`、`0.2005`、`0.2002` で、val position error は最小で `5.5550`、`5.5403`、
`5.5475` m（いずれも epoch 95 付近）だった。F1 は validation で最大 `0.8761`、`0.8764`、
`0.8749` の後に最終 `0.8719` 前後へ戻った。100 epoch まで全 run が完走し、曲線に発散や NaN は
観測されないため、学習 failure の差ではなく association 条件の差として比較できる。

一方、association failure の残存は `illegal_overlap_count=0.0` と共通である一方、duplicate active
tracks は 80.22–86.19、missed GT frames は 24.63–26.76、inactive-query false positives は
5.46–6.34 とゼロではない。したがって「ID switch が低い」だけで tracking failure が解消したとは
言えず、重複と見逃しを併記する必要がある。

### AC-006 の採用判断

AC-006 が要求する position / presence / association / convergence / failure evidence の複数観点を
比較した。許容差・指標間の重みは定義されていないため、ここでは単一の総合スコアを捏造しない。
観測値の広がり（position 5.4308771 m、F1 0.8811509、precision/recall 0.8114234/0.9638893、
missed GT 24.63）を優先し、`max_distance=0.04`, `max_missed_frames=2` の conservative を採用結論とする。
permissive の ID switches=0.60 は明確な長所として残すが、単一 seed であり、precision、duplicate
active tracks、F1、position を同時に改善していないため総合優位とは判断しない。

この結論は観測に基づく運用選択であり、deterministic association が全ての条件で因果的に優れると
いう主張ではない。次は複数 seed と AC-006 の許容差・重みを固定して再評価する。
