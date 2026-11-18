---
id: run-i1751-runtime-gpu-a3
type: run
task: tennis_scene
sequence: 55
recorded_at: '2026-11-18'
title: runtime GPU比較をCUDA異常の調査のため中断
issue: 1751
provider: codex
session: 01a1e969-a58c-7d10-be3e-4e0795fd9e18
date: '2026-11-17'
status: failed
config:
  profile: pipeline_x3d_dpt
  clip: meiji_3cam/video_000/clip_000
  cameras: 3
  source_frames: 1010
  seed: 42
  ball_batch_size: 1
  ball_precision: bf16
  compile_mode: default
metrics:
  completed_comparisons: 0
  queue_exit_code: 143
  partial_device_memory_peak_mib: 5117
  main_partial_backend_return_calls:
    court: 3
    ball: 3
    dino: 2
    appearance: 1
    vitpose: 1
repro:
  commit: ab638fb2ab825e84275863552da3f47473e390ed
  branch: codex/scene-inference-runtime
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONHASHSEED=42 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/toolchain/MultiScaleDeformableAttention.so
    .venv/bin/python -u /home/kamimura/projects/tennis-lab/.worktree-outputs/scene-inference-runtime/tennis_scene/evaluate/runtime-i1751-20261117-a3/run_pair.py
artifacts:
  run_dir: knowledge/runs/run-i1751-runtime-gpu-a3
  log: .training_queue/logs/0-8205078901338867701_2026-11-17T13-11-38-661132298Z_1380097_i1751-runtime-main-pair-20261117-a3.log
  output_dir: .worktree-outputs/scene-inference-runtime/tennis_scene/evaluate/runtime-i1751-20261117-a3
parents:
- run-i1751-runtime-gpu-a2
relations: []
papers: []
tags:
- inference-runtime
- interrupted-acceptance
- cuda-investigation
summary_impact: none
---

## 観測と中断

前2試行の計測側の問題を直し、mainとruntimeの比較を共有queueから開始した。
main側はCourt・Ball各3cameraと、人物の一部まで進んだが、完成SceneResultには到達していない。
最後のeventはCLIP-ReIDの2回目のload開始で、対応するreturnはない。
[元ログ](../../runs/run-i1751-runtime-gpu-a3/main-interrupted.log)と
[中断時点の記録](../../runs/run-i1751-runtime-gpu-a3/interruption.json)を保存する。

2026-11-18の親の再開指示によれば、CUDA異常を人間が調査するため実行を中断し、
Windowsを再起動した。queue logはexit143。再起動後のnvidia-smi応答回復は親からの報告である。
この試行のlogには、CUDAエラーの全文や最初の発生時刻は残っていない。
記録したlast_log_write時刻は最後の出力の時刻であり、CUDA異常の発生時刻とは同一視しない。
不完全なNVML samplesの最大値5117MiBから、瞬間peakや異常の原因は確定できない。

runtime側は未実行で、ペア比較・C3の受入結果は0件。部分load回数・VRAMを完成clipの指標として使わない。
prediction配列は元のoutputsに保持し、knowledgeにはコピーしていない。

## 再開時の制約

親は再起動後に再投入することを指示した。ただしCUDAエラー・ハング・nvidia-smi無応答が再発したら、
再試行を重ねず直ちに止め、全文・時刻・実行中の処理を親へ返す。
次の計測では別processの応答監視、処理heartbeat、最初の失敗後に次variantを始めない制御を用いる。
モデルのprecision・batch・推論方式を自動変更する対応はしない。

`summary_impact: none`。計測未完了で、標準pipelineの構成・X3Dの採否・精度の現行判断を更新しない。
TensorBoard学習曲線は対象外。保存scriptを再生する場合は、outputsの新しい検証先へコピーして使う。

再現入口の保存: queueが生成した原本は `queue-repro.sh` にbyte単位で保存し、
`run.json`・元command・patchも変更していない。`repro.sh` は保存bundleのscriptを新しいoutputsへ
コピーして使う追加adapterで、推論sourceの固定HEAD確認は元の `run_pair.py` が行う。
queue外からの起動は拒否する。GPU受入は停止中のため、親の再開判断前には実行しない。
