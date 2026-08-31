---
id: run-i832-blcs-tracker-conservative
type: run
title: BLCS deterministic 2D tracker conservative (#832)
issue: 832
provider: codex
session: 01a04ddb-48b6-7342-96a7-be95090cc969
date: '2026-08-30'
status: done
config:
  model: blcs_track_query
  loss: tracking_default
  data: deterministic_2d_tracking
  association:
    max_distance: 0.04
    max_missed_frames: 2
metrics:
  position_error_m: 5.4308771
  presence_f1: 0.8811509
  id_switches: 0.66
repro:
  commit: 01b6cb5f46bb6c7bdd7f4735007f84d93f1689de
  branch: feat/issue-832-deterministic-2d-tracking
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'test "$PWD" = "/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-832-deterministic-2d-tracking"
    || { echo "wrong CWD: $PWD" >&2; exit 70; }; /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_tracking paths.data_root=/home/kamimura/projects/tennis-lab/data
    run.seed=832 run.output_dir=blcs/issue-832/tracker-conservative-r3 data.augmentation.false_positive.enabled=false
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.early_stopping.enabled=false
    training.qualitative_logging.enabled=false data.association.max_distance=0.04
    data.association.max_missed_frames=2'
artifacts:
  run_dir: knowledge/runs/run-i832-blcs-tracker-conservative
  predictions: knowledge/runs/run-i832-blcs-tracker-conservative/pred_test.npz
  output_dir: outputs/blcs/issue-832/tracker-conservative-r3/logs/version_0
  curves: knowledge/runs/run-i832-blcs-tracker-conservative/curves.png
  tb_logdir: outputs/blcs/issue-832/tracker-conservative-r3/logs/version_0
parents:
- run-i832-blcs-legacy-slot-baseline
relations:
- to: run-i832-blcs-legacy-slot-baseline
  rel: compares
tags:
- blcs
- tracking-query
- deterministic-2d-tracking
- conservative
- issue-832
---

## 考察 / Findings

### 要約

旧 random-slot lifecycle の代わりに deterministic 2D association を使う conservative 設定
（`max_distance=0.04`, `max_missed_frames=2`）を、seed 832・100 epoch・FP augmentation 無効で
学習した。test は `position_error_m=5.4308771`、`presence_f1=0.8811509`、
`id_switches=0.66` で、3 候補中 position/presence と presence precision/recall、missed GT frames が最良だった。

### アーキテクチャ詳細

`blcs_track_query`（hidden dim 64、4 heads、4 stages、Q=4）の position/presence loss を用いた。
`blcs/multi_object` の同一 split、seed、100 epoch、deterministic=True、FP augmentation 無効などは
baseline と共通で、association だけを GT lifecycle の random slot packing から noisy 2D observation
ベースの deterministic tracker に変更した。tracker は距離閾値 0.04、未観測の保持は最大 2 frames、
velocity prediction 有効、再利用 gap 4 frames、overflow は error とした。

### メトリクスの解釈

test の position error は 5.4308771 m、presence F1 は 0.8811509、ID switches は 0.66 だった。
diagnostic は precision/recall=`0.8114234/0.9638893`、duplicate active tracks=`86.19`、
missed GT frames=`24.63`、inactive-query false positives=`5.56`。軸別誤差は x/y/z=`2.2662611/4.2960082/0.9529885` m で y が支配的だった。
`curves.png` では train loss が 0.6873 から 0.1827、val loss が 0.3078 から 0.2005 へ低下し、
val position error は 7.6199 から最小 5.5403（epoch 95）付近まで低下した。val F1 は最大 0.8764
（epoch 55）後に 0.8717 で、100 epoch まで有限値のまま推移し、発散や明確な崩壊は観測されない。

### アーキテクチャ⇄メトリクスの因果考察

観測として baseline 比で position error は 5.4342134→5.4308771、F1 は 0.8786702→0.8811509、
precision/recall は `0.8085196/0.9609743`→`0.8114234/0.9638893`、missed GT frames は
26.76→24.63 に改善した。一方 duplicate active tracks は 80.22→86.19、ID switches は
0.64→0.66 と悪化している。短い保持窓と厳しい距離閾値が GT の見逃しを減らす一方、query の重複発火を
増やした可能性がある、というのが仮説であり、association が因果だと単独で断定はしない。

### 既存実験との比較

親 [[run-i832-blcs-legacy-slot-baseline]]（5.4342134 m / 0.8786702 / 0.64）に対し、position と
presence は僅かに改善し、ID switches は 0.02 増えた。兄弟の permissive
[[run-i832-blcs-tracker-permissive]] は ID switches=0.60 で本 run より良いが、position=5.4355659 m、
F1=0.8787612、precision/recall=`0.8079256/0.9629848`、missed GT=25.32 で本 run を下回る。
AC-006 の許容差・重みは未定義なので、単一スコアによる優劣ではなく観測値を分けて比較する。

### 次に有効な実験

本設定を #832 の運用候補として採用し、別 seed で再現性を確認する。次は `max_distance` を 0.04
近傍で細かく振り、duplicate active tracks と ID switches の増加を抑えつつ missed GT frames を
維持できるかを検証する。複数 seed と AC-006 の評価重み・許容差を先に固定すると、今回の僅差を安全に判定できる。
