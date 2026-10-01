---
id: run-scene-component-meiji-firstframe-20260925
type: run
task: tennis_scene
sequence: 17
recorded_at: '2026-09-25'
title: Court先頭frame校正は成立、cam0人物ID容量で停止
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
  court_checkpoint_sha256: b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383
  court_temporal_policy: static_first_frame
  ball_source: external_annotation_load
  side_source: confirmed_load
  person_tracking: botsort_without_tracklet_linking
metrics:
  court_cam0_seconds: 24.596
  court_cam1_seconds: 19.884
  court_cam2_seconds: 32.472
  court_calibration_seconds: 0.038
  person_detection_cam0_seconds: 320.261
  person_detection_cam0_boxes: 2123
  person_tracking_cam0_cumulative_ids: 5
  person_tracking_camera_capacity: 4
repro:
  commit: 382005b42e2db9c5f27db252faeb8308b669d6bc
  branch: codex/clip-component-store
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/clip_components_meiji_firstframe_20260925
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-scene-component-meiji-firstframe-20260925
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790331056998359147_509730_scene-component-meiji-firstframe-20260925.log
parents:
- run-scene-component-meiji-fullclip-b863-20260925
relations: []
papers: []
tags:
- declared_components
- real_clip
- qualification
- failed
---

## 観測と判断

ユーザー指定に合わせ、Courtは各cameraのframe 0だけをKP＋LINE共同推定した。cam0、cam1、cam2の推論と初期校正は成功し、観測の出自は`static_first_frame`、全frameへの校正結果の展開は`static_first_frame_broadcast`として保存した。したがって後続frameをCourtモデルで再推論した結果ではない。3cameraの外部ball注釈も同一出力schemaのartifactとしてloadした。

cam0人物検出は1010frameで2123boxを保存した。人物trackingは累計5 IDを生成し、上限4を超えたため`person_tracking/cam0`で明示的に停止した。追跡結果のartifact、他cameraの人物検出・pose、Re-ID、side確認、3D再構成、統合scene exportはこのrunでは生成していない。失敗を回避するためのtrack棄却・slot再利用・ID統合は行っていない。

保存済み検出によるCPU診断では、遠方の同位置の人物が0–698、712–733、778–1009frameの3 IDへ分かれた可能性が高い。GMCの無効化やlost保持期間の延長だけでは累計5のままだった。score fusionを無効化すると累計4へ減るが遠方のID分裂は残るため、これだけで同一性要件を満たしたとは扱わない。全gapの本人照合は本run時点で未完了であり、後続の追跡修正・目視可視化で検証する。独立した人手3D正解はなく、Court初期校正の成功をscene精度の証拠とはしない。

推論qualificationの失敗であり、学習・TensorBoard曲線はない。正確な版・設定・evaluationはrun bundleを参照する。
