---
id: run-i983-court-throughput-b8-s42
type: run
task: court_detection
sequence: 40
recorded_at: '2026-10-12'
title: L4上のS+ batch8学習処理の速度・メモリ診断 (#983)
issue: 983
provider: codex
date: '2026-10-12'
status: done
config:
  backbone: dinov3_vits16plus
  backbone_train_mode: frozen
  transformer_dim: 1024
  transformer_depth: 8
  heads: 16
  ffn_dim: 2752
  dpt_channels: 512
  input_shape:
  - 8
  - 3
  - 288
  - 512
  batch_size: 8
  mixed_batch_counts:
  - 4
  - 4
  precision: bf16-mixed
  compile: true
  deterministic: warn
  seed: 42
  warmup_steps: 6
  measured_streaming_steps: 32
  measured_cached_steps: 12
metrics:
  streaming_seconds_per_step: 0.7953237286250214
  streaming_images_per_second: 10.058797081071164
  data_wait_seconds: 0.00041924250000136
  h2d_seconds: 0.01594727978128674
  forward_loss_seconds: 0.1683863338437135
  train_metrics_seconds: 0.03581710828126461
  backward_seconds: 0.5171598062500351
  clip_optimizer_seconds: 0.05759395796872013
  cached_images_per_second: 10.299529114004356
  peak_allocated_bytes: 9095212032
  peak_reserved_bytes: 9890168832
  streaming_gpu_utilization_percent: 92.29411764705883
artifacts:
  run_dir: knowledge/runs/run-i983-court-throughput-b8-s42
  profile: knowledge/runs/run-i983-court-throughput-b8-s42/profile.json
  config: knowledge/runs/run-i983-court-throughput-b8-s42/config.yaml
  gpu_samples: knowledge/runs/run-i983-court-throughput-b8-s42/gpu-samples.csv
  gpu_summary: knowledge/runs/run-i983-court-throughput-b8-s42/gpu-summary.json
  completion: knowledge/runs/run-i983-court-throughput-b8-s42/completion.json
  command: knowledge/runs/run-i983-court-throughput-b8-s42/request.json
  environment: knowledge/runs/group-i983-court-throughput-s42/environment.json
  input_verification: knowledge/runs/group-i983-court-throughput-s42/input-verification.json
  output_dir: gdrive:tennis_lab/outputs/court_detection/profile/i983-throughput/s42/batch08-attempt02
parents:
- run-i983-court-dinov3-splus-s42
relations:
- to: group-i983-court-throughput-s42
  rel: compared_in
papers: []
tags:
- issue-983
- colab-l4
- throughput
- diagnostic
repro:
  commit: 75e30a565b1f7a5f7fd4f08cea2daf458d377653
  command: PYTHONPATH=. .venv/bin/python "$SCRIPT_DIR/profile_throughput.py" --output
    <fresh-output> --batch-size 8
session: 01a12538-67e1-7d22-a6ce-2d7add686d59
---

L4の同一session05で、凍結DINOv3-S+と共通後段のbatch 8を短時間計測した。本学習の精度比較とは別の診断で、入力はColab VM内の `/content/tennis-lab/data`（overlay filesystem）へ取得・照合済み。合成と実画像は4:4、shapeは固定の512×288である。

元のmodel/loss/train metrics、bf16、compile、AdamW、gradient clipping、deterministic=warnを使う。6stepのwarmup後に実DataLoaderで32step、最後の実batchをGPU上で再利用して12stepを測った。Lightning logger・validation・checkpoint保存は含めず、learning rateは初期値に固定した。相別にCUDA同期を入れており、完全なTrainerや長期収束の所要時間とは区別する。TensorBoard曲線は収束実験でないため生成していない。

streamingは平均0.795324秒/step・10.0588画像/秒、GPU上のbatch再利用は10.2995画像/秒だった。データ待ちは0.419ms、GPU転送は15.947msで、今回の32stepでは入力供給の待ち時間が小さい。forward/lossは0.1684秒、train metricsは0.0358秒、backwardは0.5172秒、clip/optimizerは0.0576秒である。metrics内の多数のGPU→CPU同期はコード上存在するが、この測定では主要因と扱わない。

peak allocatedは9.095GB、reservedは9.890GBで、メモリ不足なく終了した。streaming区間のGPU使用率は92.29%（51標本）だった。GPU使用率は理論演算性能の達成率ではない。requestのsourceと保存scriptが一致し、returncode=0とDrive最終保存を確認した。

初回の診断は、追記中のGPU使用率CSVと周期Driveコピーの競合で停止した。GPU計算の失敗とは分け、元request/sourceをattemptsへ保持した。再試行では途中JSONを不変の世代ファイルとして公開し、GPU使用率CSVは収集完了後に公開する。診断の計算条件を変えず再実行した。

解釈と本学習ログとの比較は[group-i983-court-throughput-s42](000042-group-i983-court-throughput-s42.md)にまとめる。今回の4条件アブレーションのbatch8は変更していない。
