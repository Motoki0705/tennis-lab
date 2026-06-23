---
id: run-i545-s4-h4
type: run
title: i545_s4_h4
issue: 545
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-22'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.336814
  position_error_std_m: 0.316522
  position_error_median_m: 0.26532
  angular_error_deg: 8.227626
  angular_error_std_deg: 8.780316
  angular_error_median_deg: 5.937846
  x_error_m: 0.12385
  y_error_m: 0.279841
  z_error_m: 0.042334
  position_accuracy: 0.828454
  angle_accuracy: 0.871375
  position_accuracy_0.5m: 0.828454
  position_accuracy_1m: 0.981781
  position_accuracy_2m: 0.990106
  angle_accuracy_10deg: 0.727738
  angle_accuracy_15deg: 0.871375
  angle_accuracy_30deg: 0.969362
repro:
  commit: 4396ee5bcad62007b6b6ee154cd7e28d995af41b
  branch: feat/training-queue-auto-prune-ckpt
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split model.num_layers=4 model.num_task_layers=4 data=chunked_multiview_sequence_bs8
    data.batch_size=8 data.seq_len_range=[64,256] training.trainer.accumulate_grad_batches=1
    loss=canonical_rot training.trainer.max_epochs=200 training.early_stopping.enabled=false
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i545-s4-h4
  predictions: knowledge/runs/run-i545-s4-h4/pred_test.npz
  log: .training_queue/logs/1782099273737376904_481767_i545_s4_h4.log
  curves: knowledge/runs/run-i545-s4-h4/curves.png
parents:
- run-i518-exp10
relations:
- to: run-i518-exp10
  rel: compares
- to: run-i539-ex10-chunked
  rel: compares
- to: run-i539-wide-chunked
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- param-matched
---

## 考察 / Findings

### 要約
H+2S=12 パラメータ一定スイープ（77.845M）の **最良点**。共有 H=4・分岐 S=4 のバランス配分で回転 8.23°（median 5.94°）/位置 0.337m。回転・位置ともスイープ 5 本中ベスト、early-stop OFF・200ep 完遂。

### アーキテクチャ詳細
`multiview_axial_split`（`num_layers=4`=共有 trunk H、`num_task_layers=4`=rot/pose 各分岐 trunk S）。forward 総層適用 = H + 2S = 4 + 8 = 12 で EX10（H=0, S=6）と**厳密に同一パラメータ数**（77.845M, delta 0.000M）。`data=chunked_multiview_sequence_bs8`（train はチャンク生成の data-rich、val/test は固定 scene_dir で直接比較可）、effective batch=8（bs8×accum1）、`seq_len_range=[64,256]`、`loss=canonical_rot`。EX10 との唯一の差は「共有/分岐の配分」で、容量は不変。

### メトリクスの解釈
test 回転 mean 8.23°/median 5.94°、位置 mean 0.337m/median 0.265m。角度精度 `@10°=0.728 / @15°=0.871`、位置精度 `@0.5m=0.828 / @1m=0.982` でいずれもスイープ最良。curves: early-stop を切り 200ep 完遂、過学習・崩壊なし。

### アーキテクチャ⇄メトリクスの因果考察
低レベルの multiview/temporal 対応付けを共有 H=4 で束ね、回転・位置の readout を分岐 S=4 に確保する「半共有・半分岐」が、同一容量の使い道として最も効率的だった（仮説）。共有しすぎ（H 大／S 小）では分岐容量が痩せて回転・位置の競合分離が弱まり、分岐しすぎ（H 小／S 大）では低レベル特徴の共有学習が薄まる。その中間に浅い最適がある、という #545 仮説を支持する観測。

### 既存実験との比較
- パラメータ一定の親 [[run-i518-exp10]]（EX10, fully separate, 固定データ 9.98°/0.238m）に対し、**回転は本 run が上回る（8.23°<9.98°）**一方、位置は劣る（0.337>0.238m）。データ体制が違う（chunked vs 固定）ため単純な優劣ではないが、78M でも data-rich + full 学習なら回転が良化することを示す。
- [[run-i539-ex10-chunked]]（同 78M・同 chunked だが ep95 early-stop, 15.84°/0.542m）を**大幅に上回る**。差は early-stop の有無のみ＝#539 の「small は data-rich で不利」は早期終了の交絡が主因だったことを裏づける。
- 容量を倍以上に積んだ [[run-i539-wide-chunked]]（228.7M, 10.33°/0.206m）に対し、**回転は本 run が良い（8.23<10.33）**が位置は劣る（0.337>0.206）。位置は容量、回転は配分が効く、という非対称を示唆。

### 次に有効な実験
- 欠けている **S=6/H=0（fully separate）を同一プロトコル（no-early-stop, 200ep, chunked）で 1 本**取り、本 run（S=4）が真に内部最適かを確定する。
- 位置が容量律速なら、wide 容量に S=4/H=4 の配分を組み合わせた構成（広 hidden_dim × balanced trunk）で回転・位置を同時改善できるか検証。
