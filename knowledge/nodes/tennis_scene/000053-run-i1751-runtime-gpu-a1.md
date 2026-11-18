---
id: run-i1751-runtime-gpu-a1
type: run
task: tennis_scene
sequence: 53
recorded_at: '2026-11-17'
title: runtime GPU比較の初回計測失敗（profilerとcompileの干渉）
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
  ball_batch_size: 1
  ball_precision: bf16
  compile_mode: default
metrics:
  completed_comparisons: 0
  main_wall_seconds_before_failure: 184.00129055499565
  main_partial_backend_load_calls:
    court: 3
    ball: 2
  queue_exit_code: 143
repro:
  commit: ab638fb2ab825e84275863552da3f47473e390ed
  branch: codex/scene-inference-runtime
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/toolchain/MultiScaleDeformableAttention.so
    .venv/bin/python -u /home/kamimura/projects/tennis-lab/.worktree-outputs/scene-inference-runtime/tennis_scene/evaluate/runtime-i1751-20261117/run_pair.py
artifacts:
  run_dir: knowledge/runs/run-i1751-runtime-gpu-a1
  log: .training_queue/logs/0-8205081091162115878_2026-11-17T12-35-08-837884121Z_1111576_i1751-runtime-main-pair-20261117.log
  output_dir: .worktree-outputs/scene-inference-runtime/tennis_scene/evaluate/runtime-i1751-20261117
parents:
- run-scene-x3d-dpt-20261116
relations: []
papers: []
tags:
- inference-runtime
- instrumentation
- failed-acceptance
summary_impact: none
---

## 観測

#1751のモデル寿命変更を、mainと単一clipの同条件実行で比較する最初の試行。
モデルコードはmainとruntimeの固定commit、入力はMeiji 3camera clip_000の全1010frame。
人物・GVHMR・body placementを有効にし、明示X3D profileのBF16・compile(default)・batch1を維持した。
[入力と重みの内容](../../runs/run-i1751-runtime-gpu-a1/inputs.json)と
[固定した計測script](../../runs/run-i1751-runtime-gpu-a1/benchmark.py)を保持する。

main側のCourtは3回、Ballは2回loaderが呼ばれたが、2台目のBall推論で
`Observer.profile` 自身がTorch Dynamoのcompile対象になり、recompile上限8を超えて
`FailOnRecompileLimitHit: Hard failure due to fullgraph=True` で停止した。
[ログ](../../runs/run-i1751-runtime-gpu-a1/main-failure.log)と
[部分計測](../../runs/run-i1751-runtime-gpu-a1/failure-summary.json)に記録した。
この部分計測は完成clipのload回数・時間・VRAMを表さず、C3の受入比較には使わない。
runtime側の推論は開始されておらず、完成SceneResultの比較は0件。

その後、共有queue workerがTERMを受けて監督process groupを終了し、queueはexit143・
`wait=signal`になった。送信元は未確定。launch専用agentの終了記録には回収対象PIDがなく、
agentの正常終了だけを実行成功と読み替えていない。

## 解釈と次の確認

モデル設定の不一致を示す結果ではなく、Python profiler callbackとfullgraph compileの干渉である。
次の試行では計測callbackだけをcompile対象から外し、モデルのcompile・precision・batch・
入力を変えずに比較する。共有workerの寿命は一時user serviceでlaunch agentから分離する。
元の出力・queue repro・scriptは上書きしない。prediction配列はoutputsに残し、knowledgeへは移さない。
保存scriptは検証用outputsへコピーして使い、knowledge/runsの中で推論しない。

`summary_impact: none`。標準pipelineのcheckpoint、X3Dの採否、3D精度に関する現行判断を変える
証拠は得ていない。TensorBoardの学習曲線は対象外の推論受入試行である。

再現入口の保存: queueが生成した原本は `queue-repro.sh` にbyte単位で保存し、
`run.json`・元command・patchも変更していない。`repro.sh` は保存bundleのscriptを新しいoutputsへ
コピーして使う追加adapterで、推論sourceの固定HEAD確認は元の `run_pair.py` が行う。
queue外からの起動は拒否する。GPU受入は停止中のため、親の再開判断前には実行しない。
