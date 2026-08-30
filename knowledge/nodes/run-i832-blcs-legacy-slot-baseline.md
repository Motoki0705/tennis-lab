---
id: run-i832-blcs-legacy-slot-baseline
type: run
title: BLCS legacy random-slot baseline (#832)
issue: 832
provider: codex
session: 01a04ddb-48b6-7342-96a7-be95090cc969
date: '2026-08-30'
status: done
config:
  model: blcs_track_query
  loss: tracking_default
  data: legacy_lifecycle_random_slot_fp_disabled
metrics:
  position_error_m: 5.434213
  presence_f1: 0.87867
  id_switches: 0.64
repro:
  commit: 01b6cb5f46bb6c7bdd7f4735007f84d93f1689de
  branch: experiments/issue-832-legacy-baseline-boundary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'test "$PWD" = "/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-832-legacy-baseline"
    || { echo "wrong CWD: $PWD" >&2; exit 70; }; /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_tracking paths.data_root=/home/kamimura/projects/tennis-lab/data
    run.seed=832 run.output_dir=blcs/issue-832/legacy-slot-baseline-r4 data.augmentation.false_positive.enabled=false
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.early_stopping.enabled=false
    training.qualitative_logging.enabled=false'
artifacts:
  run_dir: knowledge/runs/run-i832-blcs-legacy-slot-baseline
  predictions: knowledge/runs/run-i832-blcs-legacy-slot-baseline/pred_test.npz
  output_dir: outputs/blcs/issue-832/legacy-slot-baseline-r4/logs/version_0
  curves: knowledge/runs/run-i832-blcs-legacy-slot-baseline/curves.png
  tb_logdir: .claude/worktrees/issue-832-legacy-baseline/outputs/blcs/issue-832/legacy-slot-baseline-r4/logs/version_0
parents: []
relations: []
tags:
- blcs
- tracking-query
- legacy-slot
- issue-832
---

## 考察 / Findings

### 要約

#832 の比較基準として、GT physical lifecycle を先に Q=4 へ pack し、train 時に slot をランダム置換する旧方式を seed 832・100 epoch で再学習した。test は `position_error_m=5.434213`、`presence_f1=0.878670`、post-#824 定義の `id_switches=0.64` だった。

### アーキテクチャ詳細

model は既定の `blcs_track_query`、loss は position/presence の既定構成である。data は `blcs/multi_object` の固定 split を使い、旧 `data.lifecycle.randomize_slots_train=true` のまま、比較する全 run 共通で `data.augmentation.false_positive.enabled=false` とした。invisible ball/court UV の zero-fill と MHC mixed-precision writeback の境界修正だけを候補側と共有し、association semantics は旧方式を保持した。

### メトリクスの解釈

presence は recall `0.960974` に対して precision `0.808520` で、検出を広く維持する一方、inactive query false positives `5.46` と duplicate active tracks `80.22` が残った。位置誤差は x/y/z がそれぞれ `2.272634 m`、`4.289511 m`、`0.958737 m` で、y 軸が支配的だった。`curves.png` は学習が破綻せず100 epochまで進んだことを示すが、test headlineだけでは最良epoch以後の一般化差を断定しない。

### アーキテクチャ⇄メトリクスの因果考察

観測associationをGT lifecycleから作るため、入力slotの短期連続性は人工的に安定する一方、実推論で利用できないphysical identityに依存する。高recallはこの安定性と整合する可能性があるが、duplicate active tracksとpresence precisionの低さから、ランダム置換だけではqueryの重複発火を十分抑えられていない、という仮説を置く。位置誤差の因果はcandidate比較前には確定しない。

### 既存実験との比較

このrunを #832 の新規candidate 2条件のparent baselineとする。#643/#648/#650 の保存結果は旧associationに加えてpre-#824の`id_switches`定義を含むため、数値比較には使わない。

### 次に有効な実験

同じsplit、seed、model、100 epoch、FP-disabled条件で、noise後のdeterministic 2D trackingを使う conservative (`max_distance=0.04,max_missed_frames=2`) と permissive (`0.10,8`) を完了し、position、presence、ID switch、duplicate/missed diagnostics、収束曲線をまとめて比較する。
