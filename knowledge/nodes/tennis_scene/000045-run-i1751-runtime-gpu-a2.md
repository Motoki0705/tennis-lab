---
id: run-i1751-runtime-gpu-a2
type: run
task: tennis_scene
sequence: 45
recorded_at: '2026-11-17'
title: runtime GPU比較のservice環境での計測起動失敗
issue: 1751
provider: codex
session: 01a1e969-a58c-7d10-be3e-4e0795fd9e18
date: '2026-11-17'
status: failed
config:
  profile: pipeline_x3d_dpt
  clip: meiji_3cam/video_000/clip_000
  source_frames: 1010
  cameras: 3
  seed: 42
  python_hash_seed: 42
  ball_batch_size: 1
  ball_precision: bf16
  compile_mode: default
metrics:
  completed_comparisons: 0
  model_loads_main: 0
  model_loads_runtime: 0
  queue_exit_code: 1
repro:
  commit: ab638fb2ab825e84275863552da3f47473e390ed
  branch: codex/scene-inference-runtime
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONHASHSEED=42 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/toolchain/MultiScaleDeformableAttention.so
    .venv/bin/python -u /home/kamimura/projects/tennis-lab/.worktree-outputs/scene-inference-runtime/tennis_scene/evaluate/runtime-i1751-20261117-a2/run_pair.py
artifacts:
  run_dir: knowledge/runs/run-i1751-runtime-gpu-a2
  log: .training_queue/logs/0-8205079274629829762_2026-11-17T13-05-25-370170237Z_1364572_i1751-runtime-main-pair-20261117-a2.log
  output_dir: .worktree-outputs/scene-inference-runtime/tennis_scene/evaluate/runtime-i1751-20261117-a2
parents:
- run-i1751-runtime-gpu-a1
relations: []
papers: []
tags:
- inference-runtime
- instrumentation
- failed-acceptance
summary_impact: none
---

## 観測

初回のprofiler干渉を修正してから、同じ入力・重み・推論設定で比較を試みた。
計測callbackのみを `torch.compiler.disable` の対象とし、モデルのBF16・fullgraph compile(default)・
T32・batch1は維持した。事前の[CPU smoke](../../runs/run-i1751-runtime-gpu-a2/profiler-compile-smoke.log)では、
profiler有効中にfullgraph関数を16回実行し、graph数1、synthetic loader観測1回、CUDA未初期化を確認した。

共有queueのforeground `serve` を一時user serviceが所有する構成で起動したが、
serviceのPATHにWSLの `nvidia-smi` がなく、両variantとも計測subprocessの作成で
`FileNotFoundError: nvidia-smi` となった。`orchestrator.run` に入る前で、モデルloadは両方0。
[mainログ](../../runs/run-i1751-runtime-gpu-a2/main.log)・
[runtimeログ](../../runs/run-i1751-runtime-gpu-a2/runtime.log)・
[terminal記録](../../runs/run-i1751-runtime-gpu-a2/terminal.json)を保持する。
比較用storeが無いため比較scriptも完了せず、queueはexit1。
初期measurementのrunning表示を実行中・成功とは扱わない。

## 解釈と次の確認

モデル実装・出力の証拠は得られていない。次の試行は、実在・実行権を確認した
`/usr/lib/wsl/lib/nvidia-smi` を計測scriptで明示する。モデル設定や閾値を変えて通す処置はしない。
旧試行のscript・repro・出力を上書きしない。保存scriptはoutputsの新しい検証先へコピーして実行する。

`summary_impact: none`。計測環境の起動失敗であり、標準pipelineと明示X3D profileの採否・
3D精度の判断を更新しない。TensorBoard学習曲線は対象外。

再現入口の保存: queueが生成した原本は `queue-repro.sh` にbyte単位で保存し、
`run.json`・元command・patchも変更していない。`repro.sh` は保存bundleのscriptを新しいoutputsへ
コピーして使う追加adapterで、推論sourceの固定HEAD確認は元の `run_pair.py` が行う。
queue外からの起動は拒否する。GPU受入は停止中のため、親の再開判断前には実行しない。
