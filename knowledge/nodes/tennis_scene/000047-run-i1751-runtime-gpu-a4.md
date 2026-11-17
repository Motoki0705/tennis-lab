---
id: run-i1751-runtime-gpu-a4
type: run
task: tennis_scene
sequence: 47
recorded_at: '2026-11-18'
title: mainのDINO推論でheartbeat停止を検出しGPU受入を中断
issue: 1751
provider: codex
session: 01a1e969-a58c-7d10-be3e-4e0795fd9e18
date: '2026-11-18'
status: failed
config:
  profile: pipeline_x3d_dpt
  clip: meiji_3cam/video_000/clip_000
  cameras: 3
  source_frames: 1010
  sampled_frames: 506
  seed: 42
  ball_batch_size: 1
  ball_precision: bf16
  compile_mode: default
  main_commit: 18a2fea05291eeaccb6d1dbfc81160e26c4080dd
  nvml_sampling_seconds: 1
  query_timeout_seconds: 10
  heartbeat_timeout_seconds: 180
metrics:
  completed_comparisons: 0
  queue_exit_code: 1
  main_return_code: -15
  runtime_started: false
  retries_after_anomaly: 0
  partial_device_memory_peak_mib: 3102
  nvml_samples: 585
  main_partial_backend_return_calls:
    court: 3
    ball: 3
    dino: 1
repro:
  commit: 8ef9781463e9dcd3d5a7587c1720217e276a2eaf
  branch: codex/scene-inference-runtime
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONHASHSEED=42 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/toolchain/MultiScaleDeformableAttention.so
    .venv/bin/python -u /home/kamimura/projects/tennis-lab/.worktree-outputs/scene-inference-runtime/tennis_scene/evaluate/runtime-i1751-20261118-a4/run_pair.py
artifacts:
  run_dir: knowledge/runs/run-i1751-runtime-gpu-a4
  log: .training_queue/logs/0-8205066831198893916_2026-11-17T16-32-48-801106083Z_163807_i1751-runtime-main-pair-20261118-a4.log
  output_dir: .worktree-outputs/scene-inference-runtime/tennis_scene/evaluate/runtime-i1751-20261118-a4
parents:
- run-i1751-runtime-gpu-a3
relations: []
papers: []
tags:
- inference-runtime
- hang-suspected
- failed-acceptance
summary_impact: none
---

## 観測と停止

親の再開指示に従い、Windows再起動後にmainとruntimeの比較を共有training queueへ再投入した。
main `18a2fea05` とruntime `8ef978146` を固定し、入力・重みのSHA-256は
[inputs.json](../../runs/run-i1751-runtime-gpu-a4/inputs.json)に保存した。
入力は全3camera・1010frame（30fps格子506frame、切詰めなし）。人物・GVHMR・body placementを有効にし、
seed42、X3DのBF16・compile(default)・batch1を両版で維持する条件で開始した。

**2026-11-18 01:43:07.588 JST（2026-11-17T16:43:07.588166Z）**、mainの処理heartbeatが
180秒更新されず、独立監視processが `worker_heartbeat_timeout` を記録した。
最後のheartbeatは01:40:06.610 JSTで、最後に観測した処理は
`_tennis_lab_third_party_dino_models.dino.deformable_transformer.DeformableTransformer.forward`、
`third_party/DINO/models/dino/deformable_transformer.py:289` の `c_return` だった。
この位置は最後の観測点であり、実際に停止したstackの位置と断定しない。

[異常記録の全文](../../runs/run-i1751-runtime-gpu-a4/anomaly.json)、
[停止報告](../../runs/run-i1751-runtime-gpu-a4/failure-report.json)、
[mainの元ログ](../../runs/run-i1751-runtime-gpu-a4/main.log)、
[監視結果](../../runs/run-i1751-runtime-gpu-a4/health-main.json)を保存した。
main PID164183を終了させreturncode=-15、queueはfailed・exit1となった。
`termination_pending=false` を確認し、最初の失敗後のruntime起動は行っていない。
queueの次受付も停止し、異常検出後の再試行は0回。

## 判明したことと限界

NVMLは01:43:06.587 JSTまで応答し、driver621.24を返していた。
明示的なCUDA例外文はログにない。SIGUSR1によるfaulthandler出力は取得できず、停止stackも未取得。
したがって、今回はheartbeat停止に基づく**ハング疑い**であり、CUDA・driverの異常とは断定できない。
計測callbackの不具合・検出の誤りも除外できていない。親の停止条件に従って再投入を止め、原因確認と
次のGPU実行の判断を親へ返す。PR #2182はDraftを維持し、Ready for reviewにはしていない。

完成SceneResult比較は0件。runtimeの実モデルload/unload・peak VRAMは未確認。
mainの部分loader returnはCourt3・Ball3・DINO1回、NVML585sampleの観測最大値は3102MiBだったが、
完成clipのモデル読込回数やpeak VRAMの受入指標として使わない。
prediction配列・NVML生samples・profiler eventは元outputsに保持し、knowledgeへコピーしていない。

## CPU検証と再現記録

main統合後の関連CPUテストは354 passed / 37.70秒、通常pre-commitは成功。
監視側のCPU smoke9項目は成功した（応答期限・失敗全文・heartbeat停止・main失敗後のruntime未起動・
profilerのcompile非干渉等）。[CPUテストlog](../../runs/run-i1751-runtime-gpu-a4/cpu-tests.log)と
[smoke log](../../runs/run-i1751-runtime-gpu-a4/guard-smoke.log)を保存する。
推論source HEAD `8ef978146` の[CIは全成功](https://github.com/Motoki0705/tennis-lab/actions/runs/44122067520)。

計測script・入力pin・6ファイルの固定hash・queueの原再現bundleを保存した。
callbackはCUDA APIを呼ばず、NVMLは別process・1秒間隔、query期限10秒、処理heartbeat期限180秒だった。
これらは停止を検出する運用条件で、GPU実行の費用予算ではない。
保存scriptは当時のworktreeとoutputsを参照する歴史的記録である。再開判断が出るまでは実行しない。
再生する場合も新しい検証用outputsへコピーし、knowledge/runs内では推論しない。

`summary_impact: none`。受入未完了であり、標準pipelineの構成・X3Dの採否・3D精度の現行判断を更新しない。
学習ではないためTensorBoard学習曲線は対象外。

再現入口の保存: queueが生成した原本は `queue-repro.sh` にbyte単位で保存し、
`run.json`・元command・patchも変更していない。`repro.sh` は保存bundleのscriptを新しいoutputsへ
コピーして使う追加adapterで、推論sourceの固定HEAD確認は元の `run_pair.py` が行う。
queue外からの起動は拒否する。GPU受入は停止中のため、親の再開判断前には実行しない。
