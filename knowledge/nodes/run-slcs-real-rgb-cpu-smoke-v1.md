---
id: run-slcs-real-rgb-cpu-smoke-v1
type: run
title: '実RGB全体版のproduction window計数と1バッチCPU学習スモーク'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/real_rgb_v1
  profile: train_real_rgb
  dry_run: true
  batch_size: 16
  device: cpu
metrics:
  train_windows: 466
  val_windows: 343
  test_windows: 239
  incomplete_clips: 0
  dropped_low_quality_windows: 0
  full_training_steps_per_epoch: 30
  full_training_steps_at_60_epochs: 1800
  smoke_batches: 1
  smoke_elapsed_seconds: 135.09
  smoke_max_rss_kib: 18370148
  smoke_exit_code: 0
repro:
  commit: a4655661
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/time -v
    .venv/bin/python -m src.tasks.slcs.scripts.train --config-name train_real_rgb
    run.dry_run=true run.output_dir=slcs/train/real_rgb_cpu_smoke/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-real-rgb-cpu-smoke-v1
  output_dir: outputs/slcs/train/real_rgb_cpu_smoke/s42-takeover-001
parents: [run-slcs-real-rgb-full-assembly-v1]
tags: [slcs, real-rgb, cpu, smoke, dataset]
---

## 考察 / Findings

### 要約

全体版の本番loaderが全61clipを読み込み、1バッチのCPU forward/backward・更新を完了した。
通常の60epoch学習や精度評価ではない。事前のread-only計数ではtrain/val/testが466/343/239窓だった。

### アーキテクチャ詳細

`train_real_rgb` を変更せず、run.dry_runと出力先だけを指定。dry-run runnerはCPU・num_workers=0、
1batch、checkpoint/logger無効で実行する。元profileのbatch16、120frame、品質設定、DINO必須は維持する。
モデルは学習可能約2.7M parameter。batchのDINO形状は(16,12,448,768)、playerは(16,2,120,17,2)。

### メトリクスの解釈

計数はscoutによる同じ本番configのCPU事前検証で、各splitを順次構築・破棄した結果。
実行した完全なcommandとstdoutを添付し、上限推定と区別する。clip欠落・低品質window除外はいずれも0。
CPU smokeは135.09秒、最大RSS 18370148KiB、swap 0、終了コード0。性能や汎化の証拠ではない。

### アーキテクチャ⇄メトリクスの因果考察

RGB特徴を含む実データ・入力adapter・モデル・lossの接続が1batchで成立した。
CPU runnerはデータ構築を繰り返すため、このRSSをGPU学習のVRAMやworker数倍のメモリと解釈しない。

### 既存実験との比較

旧pilotは88train windows・6更新/epochだった。全体版は30更新/epoch、60epochで1800更新となり、
warmup200は約11.1%。dataset版・更新回数も違うため、pilotとの性能差を単一loss施策の効果と扱わない。

### 次に有効な実験

共有queueで単独loss比較と全体版60epoch学習を行う。validation選定重みに対し、両domainのheld-out誤差と
入力条件差・平均位置baseline・motionを評価する。本smokeはTensorBoard学習曲線を生成しない。
