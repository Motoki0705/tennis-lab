---
id: run-i832-blcs-tracker-permissive
type: run
title: BLCS deterministic 2D tracker permissive (#832)
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
    max_distance: 0.10
    max_missed_frames: 8
metrics:
  position_error_m: 5.4355659
  presence_f1: 0.8787612
  id_switches: 0.6
repro:
  commit: 103d70eb8c8b8857dd9ba3936d53084c5d24bc8d
  branch: feat/issue-832-deterministic-2d-tracking
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'test "$PWD" = "/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-832-deterministic-2d-tracking"
    || { echo "wrong CWD: $PWD" >&2; exit 70; }; /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_tracking paths.data_root=/home/kamimura/projects/tennis-lab/data
    run.seed=832 run.output_dir=blcs/issue-832/tracker-permissive-r3 data.augmentation.false_positive.enabled=false
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.early_stopping.enabled=false
    training.qualitative_logging.enabled=false data.association.max_distance=0.10
    data.association.max_missed_frames=8'
artifacts:
  run_dir: knowledge/runs/run-i832-blcs-tracker-permissive
  predictions: knowledge/runs/run-i832-blcs-tracker-permissive/pred_test.npz
  output_dir: outputs/blcs/issue-832/tracker-permissive-r3/logs/version_0
  curves: knowledge/runs/run-i832-blcs-tracker-permissive/curves.png
  tb_logdir: outputs/blcs/issue-832/tracker-permissive-r3/logs/version_0
parents:
- run-i832-blcs-legacy-slot-baseline
relations:
- to: run-i832-blcs-legacy-slot-baseline
  rel: compares
tags:
- blcs
- tracking-query
- deterministic-2d-tracking
- permissive
- issue-832
---

## 考察 / Findings

### 要約

deterministic 2D association の permissive 設定（`max_distance=0.10`, `max_missed_frames=8`）を、
他候補と同じ seed 832・100 epoch・FP augmentation 無効で学習した。test は
`position_error_m=5.4355659`、`presence_f1=0.8787612`、`id_switches=0.60` で、ID switches は
3 候補中最良だが、総合採用は conservative に譲る。

### アーキテクチャ詳細

`blcs_track_query`（hidden dim 64、4 heads、4 stages、Q=4）の position/presence loss と
`blcs/multi_object` の固定 split を使った。seed、100 epoch、deterministic=True、FP augmentation
無効、noise/dropout augmentation は conservative と共通で、difference は noisy 2D association の
距離閾値 0.10 と未観測保持 8 frames である。velocity prediction、再利用 gap 4 frames、mean cost、
overflow error は共通で、旧 baseline の random slot lifecycle から deterministic tracking へ移行した。

### メトリクスの解釈

test の position error は 5.4355659 m、presence F1 は 0.8787612、ID switches は 0.60 だった。
diagnostic は precision/recall=`0.8079256/0.9629848`、duplicate active tracks=`83.53`、
missed GT frames=`25.32`、inactive-query false positives=`6.34`。軸別誤差は x/y/z=`2.2656117/4.2986570/0.9564066` m で、他候補と同じく y が最大だった。
`curves.png` では train loss が 0.6872 から 0.1823、val loss が 0.3061 から 0.2002 へ低下し、
val position error は 7.6009 から最小 5.5475（epoch 95）付近まで下がった。val F1 は最大 0.8749
（epoch 20）後は 0.8718 近傍で、100 epoch の曲線に発散や NaN は観測されないが、後半は頭打ちである。

### アーキテクチャ⇄メトリクスの因果考察

観測として permissive は baseline 比で ID switches を 0.64→0.60 に下げ、missed GT frames も
26.76→25.32 に減らした。一方 precision は 0.8085196→0.8079256、duplicate active tracks は
80.22→83.53、inactive-query false positives は 5.46→6.34 となり、距離・保持を緩めたことと
整合する false-positive 側の悪化がある。長い保持窓が同一物体の再接続を助け ID switches を減らした
可能性は仮説であり、今回の単一 seed から一般化はできない。

### 既存実験との比較

親 [[run-i832-blcs-legacy-slot-baseline]]（5.4342134 m / 0.8786702 / 0.64）と比べ、ID switches は
改善したが position error は 0.0013529 m、F1 は 0.0000911 悪化した。兄弟の conservative
[[run-i832-blcs-tracker-conservative]] は position=5.4308771 m、F1=0.8811509、
precision/recall=`0.8114234/0.9638893`、missed GT=24.63 で、ID switches 以外の主要観点で上回る。
AC-006 の許容差・重みは未定義のため、ID switches 最良だけを採用理由にはしない。

### 次に有効な実験

permissive は association の再接続能力を確認する比較点として保存する。次は複数 seed で ID
switches の改善が再現するか確認し、同時に false positives・duplicate tracks を抑える cost または
保持窓の制約を検証する。採用候補は conservative を軸に、0.04 から 0.10 の中間閾値を評価する。
