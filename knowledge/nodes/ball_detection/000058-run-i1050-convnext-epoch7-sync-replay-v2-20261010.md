---
id: run-i1050-convnext-epoch7-sync-replay-v2-20261010
type: run
task: ball_detection
sequence: 58
recorded_at: '2026-10-10'
title: ConvNeXt V2の失敗prefixを同期CUDAで16更新再生
issue: 1050
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-10'
status: done
config:
  model: convnext_v2/factorized
  precision: bf16
  checkpoint_step: 42000
  launch_blocking: true
  phase_synchronization: true
  updates: 16
metrics:
  replayed_updates: 16
  last_replayed_step: 42016
  seconds: 22.52211118300329
repro:
  commit: 0e2a12c46a46f910df4da6885179c45cfca8b5f6
  branch: codex/ball-cuda-replay-20261010
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/ball-cuda-replay-20261010
    CUDA_LAUNCH_BLOCKING=1 CUDA_LOG_FILE=stderr OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    tests/benchmarks/ball_dpt_checkpoint_replay.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --checkpoint /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/mdd-cnn-comparison/convnext_v2-s42-v2-serial/epoch-006.pt
    --failure /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/mdd-cnn-comparison/convnext_v2-s42-v2-serial/failure.json
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/mdd-cnn-recovery/20261010-sync-replay-v2
    --updates 16
artifacts:
  run_dir: knowledge/runs/run-i1050-convnext-epoch7-sync-replay-v2-20261010
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791594302566371947_2622790_i1050-convnext-epoch7-sync-replay-v2-20261010.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/mdd-cnn-recovery/20261010-sync-replay-v2
parents:
- run-i1050-convnext-v2-s42-u60000
relations: []
papers: []
tags:
- cuda-diagnostic
- checkpoint-replay
- bf16
---

ConvNeXt V2のserial nvJPEG本学習は、6000更新診断を通過した後、42000更新のcheckpointを保存し、42004更新目でCUDA unknown errorが表面化した。先読み無効化だけでは長期障害を解消していない。同じcheckpoint・optimizer・RNG・sampler epoch 7のprefixを使い、失敗batchのclip/start/frame_step/frame_indicesが4番目の更新に一致することを確認した。

GPU診断ではCUDA_LAUNCH_BLOCKING=1に加え、decode/forward/loss/backward/grad norm/optimizerの各段階で同期した。BF16・compile default・nvJPEGを維持し、16更新（42001〜42016）はfinite loss/gradで正常終了した。診断は本学習checkpointを更新せず、重みを保存・転送していない。JPEG bytesは対象clipのhash/GT検証をしてCPUへ先読みし、DataLoader workerは使わない。この入出力スケジュールと短いprocess寿命が本学習と異なる。

失敗batchを含むprefixで再現しなかったという観測であり、同期化が根本原因を修正した証明ではない。次は同期起動設定をresume_receiptsへ明示して、元checkpointから残予算だけ学習し長時間の挙動を確認する。residualは30000、FasterNetは18000、ConvNeXt V2は42000の保存済み状態をCPUで検証済み。精度・モデル・データ・seed・学習率は維持するが、処理速度を同期前と単純比較しない。

最初の診断投入はtests.benchmarksのmodule解決でGPU起動前に失敗した。tracked scriptを明示しPYTHONPATHを固定したv2で上記の診断が完了した。TensorBoard・test精度・追加のモデル比較はこの診断では取得しない。
