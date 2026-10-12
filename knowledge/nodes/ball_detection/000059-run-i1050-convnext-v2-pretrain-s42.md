---
id: run-i1050-convnext-v2-pretrain-s42
type: run
task: ball_detection
sequence: 59
recorded_at: '2026-10-11'
title: ConvNeXt V2＋DPT事前学習6万更新完了・ユーザー選択encoder
issue: 1050
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-10'
status: done
config:
  model:
    stem_channels:
    - 16
    - 32
    - 64
    - 128
    mixed_channels:
    - 192
    - 256
    residual_blocks:
    - 0
    - 1
    - 2
    - 2
    - 1
    - 1
    decoder_channels: 128
    dim: 256
    heads: 8
    layers: 4
    ffn_dim: 704
    dropout: 0.1
    rope_base: 10000.0
    activation_checkpointing: false
    encoder_variant: convnext_v2
    temporal_mixing: factorized
  precision: bf16
  seed: 42
  batch_size: 1
  epochs: 10
  windows_per_epoch: 6000
  manifest_sha256: 035d3ab96807ace8e25ebe3a5d603f09a7c514802942170fdd5b8ad2854c949c
  selection_scope: common
  test_usage: none
metrics:
  completed_updates: 60000
  best_step: 48000
  common_mean_error_px: 17.34708861378516
  full_mean_error_px: 20.012687514479662
repro:
  commit: 5f8746b49cee4af6f5c4155105c259893ff32c1e
  branch: HEAD
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_LAUNCH_BLOCKING=1 CUDA_LOG_FILE=stderr OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
    PYTHONUNBUFFERED=1 /home/kamimura/projects/tennis-lab/.claude/worktrees/ball-cnn-campaign-serial-20261009/.venv/bin/python
    -m src.tasks.ball_detection.scripts.pretrain_mdd_dpt --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/.claude/worktrees/ball-cnn-campaign-serial-20261009/src/tasks/ball_detection/configs/model/mdd_dpt_convnext_v2.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/mdd-cnn-comparison/convnext_v2-s42-v2-serial
    --epochs 10 --windows-per-epoch 6000 --learning-rate 0.0002 --warmup-updates 500 --seed 42 --device cuda --precision
    bf16 --batch-size 1 --num-workers 8 --prefetch-factor 4 --cpu-threads 2 --pin-memory --jpeg-decoder nvjpeg --input-verification
    upfront --compile-mode default --selection-scope common --sigma-ratio 0.012 --focal-gamma 2.0 --log-every 50
    --preview-clips 3 --resume /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/mdd-cnn-comparison/convnext_v2-s42-v2-serial/epoch-006.pt
artifacts:
  run_dir: knowledge/runs/run-i1050-convnext-v2-pretrain-s42
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791594512988227844_2629034_i1050-convnext-v2-resume-e7-blocking-20261010.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/mdd-cnn-comparison/convnext_v2-s42-v2-serial
  checkpoint: epoch-007.pt
  checkpoint_sha256: 73f1f0d50693f4d6a84ba451f08cc1918d8fb1a0483a60a97874d916a249745f
parents:
- run-i1050-convnext-epoch7-sync-replay-v2-20261010
relations: []
papers: []
tags:
- mdd
- convnext-v2
- bf16
- single-seed
- validation-only
---

ConvNeXt V2系・factorized temporal CNN＋DPTの事前学習を合計60,000更新まで完了した。検証で選択されたのは48,000更新の `epoch-007.pt`。Common平均位置誤差17.3471px、Full20.0127pxとなった。ユーザーがこの完了済みCNNを事後学習のencoderとして明示的に選択した。residual/FasterNetは途中で取消され、3候補を完走した比較の勝者ではない。

## 条件と経過

- 固定manifestのtrain824 clips、validation190 clips（Common56 clips）。入力は720×1280・32frame、元/1/2/1/4 FPSを混合し、間引いたRGBからFP32 MDDを生成する。学習計算BF16、BS1、seed42。
- DPTの1/4解像度heatmapを位置教師で学習。AdamW、peak LR2e-4、warmup500、cosine、6,000窓×10epochs。詳細はbundleの `config.json` が正本。
- serial nvJPEGのrunは42,000更新checkpointを残してCUDA異常終了した。同じ失敗prefixを同期実行で確認してから、`CUDA_LAUNCH_BLOCKING=1` の再開で60,000更新に到達した。bundleの再現コマンドは42,000更新からの最終再開を記録し、初期化からのコマンドではない。事前に存在する `epoch-006.pt` が必要。
- 元のCUDA障害の根本原因は未確定。この完了1件から一般的な長期安定性やCNN間の速度差を断定しない。baseline/FasterNetの残予算は実行しない。

## 選択と検証

選択指標はCommonの各FPS平均誤差の等重み平均。単位は元画像pixelで、入力リサイズ後のpixelやF1ではない。Fullは選択に使わず、testは未評価。`best.json`・checkpoint・完了marker・凍結manifestの一致を事後学習の完了検証でも再確認した。親checkpoint SHA-256は `73f1f0d50693f4d6a84ba451f08cc1918d8fb1a0483a60a97874d916a249745f`。

次段の実測と採否は子ノード `run-i1049-convnext-query-posttrain-s42` を参照。TensorBoard曲線はなく、保存済み `metrics.jsonl` と `train-summary.jsonl` を証拠とする。後者は学習ログの数値を保持した抜粋で、checkpoint自体はローカル出力に残した。
