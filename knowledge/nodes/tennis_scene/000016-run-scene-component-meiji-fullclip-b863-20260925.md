---
id: run-scene-component-meiji-fullclip-b863-20260925
type: run
task: tennis_scene
sequence: 16
recorded_at: '2026-09-25'
title: Court全frame推論を先頭frame方式への指示変更で中止
issue: 915
provider: codex
session: 01a0d0f9-6057-7961-ad6b-adc46744d619
date: '2026-09-25'
status: failed
config:
  pipeline: declared_components_v1
  clip: video_000/clip_000
  frames: 1010
  court_checkpoint_sha256: b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383
  court_temporal_policy: per_frame
  queue_status: cancelled
metrics: {}
repro:
  commit: 93f6035e0733acedfa61940cd78faf2055fd4aee
  branch: codex/clip-component-store
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/clip_components_meiji_b863_20260925
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-scene-component-meiji-fullclip-b863-20260925
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790328751614279345_308143_scene-component-meiji-fullclip-b863-20260925.log
parents:
- run-scene-component-meiji-fullclip-r2-20260925
relations: []
papers: []
tags:
- declared_components
- real_clip
- qualification
- cancelled
---

## 中止理由

以前の成功記録と同じSHAのCourt checkpoint b863…を明示して再実行した。cam0の領域探索を通過して全frame推論を進めたが、約29分の時点でユーザーから「court推論は最初の1フレームのみで大丈夫」と指示があり、共有queueで実行を中止した。数値エラー・モデル失敗による中断ではない。

Court componentの完了成果物はまだ公開されておらず、全pipelineの完走・精度を示すmetricはない。外部ball成果物3件は保存済み。記録schemaにcancelled状態がないためstatus=failedとし、実際のqueue状態をconfigに記録した。

次の版ではCourtの観測をframe 0だけに限定し、固定cameraの幾何を全frameへ明示的に展開する。KP＋LINE共同推定は変更しない。学習runではなくTensorBoard曲線はない。
