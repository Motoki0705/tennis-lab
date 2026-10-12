---
id: run-i983-court-throughput-b16-s42
type: run
task: court_detection
sequence: 41
recorded_at: '2026-10-12'
title: L4上のS+ batch16学習処理の速度・メモリ診断 (#983)
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
  - 16
  - 3
  - 288
  - 512
  batch_size: 16
  mixed_batch_counts:
  - 8
  - 8
  precision: bf16-mixed
  compile: true
  deterministic: warn
  seed: 42
  warmup_steps: 6
  measured_streaming_steps: 32
  measured_cached_steps: 12
metrics:
  streaming_seconds_per_step: 1.5438460487500691
  streaming_images_per_second: 10.363727661157629
  data_wait_seconds: 0.00044689412507636916
  h2d_seconds: 0.032300753593716536
  forward_loss_seconds: 0.33801735309377534
  train_metrics_seconds: 0.06557085590621625
  backward_seconds: 1.0395770723749962
  clip_optimizer_seconds: 0.06793311965628845
  cached_images_per_second: 10.676559810175018
  peak_allocated_bytes: 16157158400
  peak_reserved_bytes: 17450401792
  streaming_gpu_utilization_percent: 93.38383838383838
artifacts:
  run_dir: knowledge/runs/run-i983-court-throughput-b16-s42
  profile: knowledge/runs/run-i983-court-throughput-b16-s42/profile.json
  config: knowledge/runs/run-i983-court-throughput-b16-s42/config.yaml
  gpu_samples: knowledge/runs/run-i983-court-throughput-b16-s42/gpu-samples.csv
  gpu_summary: knowledge/runs/run-i983-court-throughput-b16-s42/gpu-summary.json
  completion: knowledge/runs/run-i983-court-throughput-b16-s42/completion.json
  command: knowledge/runs/run-i983-court-throughput-b16-s42/request.json
  environment: knowledge/runs/group-i983-court-throughput-s42/environment.json
  input_verification: knowledge/runs/group-i983-court-throughput-s42/input-verification.json
  output_dir: gdrive:tennis_lab/outputs/court_detection/profile/i983-throughput/s42/batch16
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
  commit: c4b12f923b243da6b502149aa6f3912c706045e7
  command: PYTHONPATH=. .venv/bin/python "$SCRIPT_DIR/profile_throughput.py" --output
    <fresh-output> --batch-size 16
session: 01a12538-67e1-7d22-a6ce-2d7add686d59
---

L4の同一session05で、凍結DINOv3-S+と共通後段のbatch 16を短時間計測した。本学習の精度比較とは別の診断で、入力はColab VM内の `/content/tennis-lab/data`（overlay filesystem）へ取得・照合済み。合成と実画像は8:8、shapeは固定の512×288である。

元のmodel/loss/train metrics、bf16、compile、AdamW、gradient clipping、deterministic=warnを使う。6stepのwarmup後に実DataLoaderで32step、最後の実batchをGPU上で再利用して12stepを測った。Lightning logger・validation・checkpoint保存は含めず、learning rateは初期値に固定した。相別にCUDA同期を入れており、完全なTrainerや長期収束の所要時間とは区別する。TensorBoard曲線は収束実験でないため生成していない。

streamingは平均1.543846秒/step・10.3637画像/秒、GPU上のbatch再利用は10.6766画像/秒だった。データ待ちは0.447ms、GPU転送は32.301msで、今回の32stepでは入力供給の待ち時間が小さい。forward/lossは0.3380秒、train metricsは0.0656秒、backwardは1.0396秒、clip/optimizerは0.0679秒である。metrics内の多数のGPU→CPU同期はコード上存在するが、この測定では主要因と扱わない。

peak allocatedは16.157GB、reservedは17.450GBで、メモリ不足なく終了した。streaming区間のGPU使用率は93.38%（99標本）だった。GPU使用率は理論演算性能の達成率ではない。requestのsourceと保存scriptが一致し、returncode=0とDrive最終保存を確認した。

解釈と本学習ログとの比較は[group-i983-court-throughput-s42](000042-group-i983-court-throughput-s42.md)にまとめる。今回の4条件アブレーションのbatch8は変更していない。
