---
id: run-issue-753-track-query-baseline-matched-v4
type: run
title: issue-753-track-query-baseline-matched-v4
issue: 753
provider: codex
session: 01a02003-4b40-79c3-abbf-24c54b29c0dc
date: '2026-08-21'
status: done
config:
  model: track-query-h64-q4-stage4-mhc-hybrid-cswa-reference
  loss: position+presence; smoothness=0; gravity=0
  data: baseline-matched synthetic; seed=643648; V=2-3; T=8-12; P=1-3; split=96/24/24
metrics:
  loss: 0.206459
  loss_position: 0.024598
  loss_position_x: 0.023273
  loss_position_y: 0.033974
  loss_position_z: 0.008496
  loss_presence: 0.181861
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 0.331892
  presence_precision: 0.72928
  presence_recall: 0.967118
  presence_f1: 0.831521
  lifecycle_presence_f1: 0.831521
  birth_frame_error: 0.126374
  death_frame_error: 0.340659
  query_reuse_count: 0.0
  illegal_overlap_count: 0.0
  segment_id_switches: 17.333334
  id_switches: 17.333334
  duplicate_active_tracks: 1.666667
  missed_gt_frames: 3.333333
  inactive_query_false_positives: 32.0
  position_mae_x_m: 0.979123
  position_mae_y_m: 2.546809
  position_mae_z_m: 0.110009
repro:
  commit: 296310e8e6749079bbbe60ac68329b8adb31b22a
  branch: feat/blcs-track-query-cswa
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: .venv/bin/python -m experiments.blcs.issue_753.scripts.train
artifacts:
  run_dir: knowledge/runs/run-issue-753-track-query-baseline-matched-v4
  predictions: knowledge/runs/run-issue-753-track-query-baseline-matched-v4/pred_test.npz
  log: .training_queue/logs/1787251520739080033_2114469_issue-753-track-query-baseline-matched-v4.log
  output_dir: outputs/blcs/issue-753-baseline-matched/logs/version_3
  curves: knowledge/runs/run-issue-753-track-query-baseline-matched-v4/curves.png
  tb_logdir: outputs/blcs/issue-753-baseline-matched/logs/version_3
parents:
- run-issue-648-multiball-baseline
relations: []
tags:
- blcs
- multi-ball
- tracking
- mhc
- cswa
- synthetic
---

## 考察 / Findings

### 要約

現行のfixed-query BLCSモデルを、`run-issue-648-multiball-baseline` と同じseed `643648`、split `96/24/24`、`V=2-3`、`T=8-12`、ball数 `1-3`、batch 8、5 epoch、LR `3e-4`、float32の規模へ合わせて学習した。testでは `position_error=0.331892`、`presence_f1=0.831521`、`id_switches=17.333334` となった。学習・test prediction保存・knowledge bundle昇格まで完走した。

### アーキテクチャ詳細

モデルはhidden 64、4 heads、FFN 128、4 persistent queries、4 stages、約590K parametersである。各stageは4 query streamをmHCで混合し、camera token temporal block、同一時刻のspatial block、query temporal blockを適用する。stage cycleは固定の `C,C,C,G` で、先頭3 stageのtemporal blockはcompression ratio 4・window radius 4のreference CSWA、最後のstageはglobal MHAを使う。全blockのFFNはSwiGLUで、spatial座標にはtime/camera/role RoPE、temporal処理にはtime RoPEを使う。lossはpositionとpresenceのみで、smoothness/gravity weightは0である。

データはIssue #648の決定論的synthetic recipeの規模と乱数条件を、現行の `(B,V,T,Q,2)` fixed-width lifecycle contractへ変換した実験専用datasetである。観測slotと教師slotはtrain時に独立permuteし、dropout `0.12`、false positive probability `0.45`、UV noise std `0.008` を適用した。

### メトリクスの解釈

position errorは `0.331892 m` で、presenceはprecision `0.729280`、recall `0.967118`、F1 `0.831521` だった。高recallに対してprecisionが低く、`missed_gt_frames=3.333333` を抑える代わりに `inactive_query_false_positives=32.0` と `duplicate_active_tracks=1.666667` が残った。`id_switches=17.333334`、`segment_id_switches=17.333334` であり、identity維持はbaselineから改善していない。

### アーキテクチャ⇄メトリクスの因果考察

観測として、位置誤差は小さい一方、presenceは過検出側へ偏った。仮説として、fixed-widthの4 observation slotsと4 persistent queries、mHCによるquery stream混合は存在中の軌道情報を保持してrecallを上げやすいが、inactive queryを十分抑制できずfalse positiveとduplicateを増やした可能性がある。ただし、CSWA/mHCへの変更とdataset contract変換が同時に入っているため、この1 runだけから個別コンポーネントの因果効果は断定できない。

### 既存実験との比較

親baselineの `position_error=0.337709` に対し `0.331892` で、差は `-0.005817 m`（相対 `-1.72%`）だった。`presence_f1` は `0.847570` から `0.831521` へ `-0.016049`、`id_switches` は両者とも `17.333334` である。presenceの内訳はprecisionが `0.809143` から `0.729280` へ低下し、recallが `0.890765` から `0.967118` へ上昇した。これに対応してmissed GTは `9.666667` から `3.333333` へ減ったが、inactive-query false positiveは `9.666667` から `32.0`、duplicate active tracksは `0.333333` から `1.666667` へ増えた。

比較規模、seed、主要な学習条件は合わせたが、完全な同条件比較ではない。baselineは2-stage M-RoPE attentionと最大6 candidateの旧unordered-candidate generator、今回のモデルは4の倍数を要求する4-stage mHC/CSWAとcandidate width 4の現行fixed-Q contractである。また単一seedなので、`5.8 mm` のposition差を統計的改善とは判断しない。

### 次に有効な実験

まず同じ実験専用datasetを固定して3 seedsを実行し、position差とpresenceのprecision/recall trade-offの再現性を確認する。その上で、stage数を4に揃えたglobal-attention対照と `C,C,C,G` を比較し、CSWAの効果をdataset差から分離する。presence過検出については `presence_inactive_weight` とmatching presence weightを単独で振り、`presence_f1`、inactive-query false positive、duplicate active tracksを主要判定指標にする。
