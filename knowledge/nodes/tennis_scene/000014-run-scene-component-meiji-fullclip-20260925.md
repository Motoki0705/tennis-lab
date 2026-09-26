---
id: run-scene-component-meiji-fullclip-20260925
type: run
task: tennis_scene
sequence: 14
recorded_at: '2026-09-25'
title: 設定解釈で停止した宣言型pipelineの初回起動
issue: 915
provider: codex
session: 01a0d0f9-6057-7961-ad6b-adc46744d619
date: '2026-09-25'
status: failed
config:
  pipeline: declared_components_v1
  clip: video_000/clip_000
  frames: 1010
  cameras: 3
  ball_source: external_annotation_load
  side_source: confirmed_load
metrics: {}
repro:
  commit: 9b8f7d3cd2ead11b0e9c339049fe8d4af1666da4
  branch: codex/clip-component-store
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/clip_components_meiji_20260925
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-scene-component-meiji-fullclip-20260925
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790327924714558768_288123_scene-component-meiji-fullclip-20260925.log
parents: []
relations: []
papers: []
tags:
- declared_components
- real_clip
- qualification
- failed
---

## 結果と次の判断

実clip全体検証の初回起動は、Court checkpointファイル名の`epoch=17`をHydra overrideで引用していなかったため、設定解釈で停止した。モデル推論・clip成果物生成には到達しておらず、精度や処理時間の評価値はない。

checkpoint文字列を引用し、role-relative checkpoint pathを使うよう検証スクリプトを修正した。既存Re-IDのheadless exportをCHECKPOINT rootへ同じSHAのまま配置した。モデルや重みは学習・変更していない。構成の読み込みをCPUで確認してから次の全体検証へ進めた。

学習runではなく設定preflightの失敗なのでTensorBoard・学習曲線はない。失敗した正確なコード版・コマンドはrun bundleを正本とする。
