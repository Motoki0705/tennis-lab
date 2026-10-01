---
id: run-scene-component-meiji-tracklets-20260925
type: run
task: tennis_scene
sequence: 18
recorded_at: '2026-09-25'
title: tracklet短期欠落の結合後、宣言型pipelineを初めて完走
issue: 915
provider: codex
session: 01a0d81f-1a8d-7462-b0f8-d827f54c6a4a
date: '2026-09-25'
status: done
config:
  pipeline: declared_components_v1
  clip: video_000/clip_000
  frames: 1010
  cameras: 3
  court_temporal_policy: static_first_frame
  person_tracks_schema: 2
  person_tracking: botsort_then_unique_short_gap_links
  ball_source: external_annotation_load
  side_source: geometry_confirmed_load
  confirmed_half_turns: [false, false, true]
metrics:
  ball_3d_valid_frames: 987
  player_global_ids: 4
  player_joint_valid_frames: [1010, 0, 26, 50]
  player_root_valid_frames: [815, 0, 25, 0]
  player_smpl_valid_frames: [780, 0, 25, 0]
  cam0_stable_local_ids: 3
  cam1_stable_local_ids: 2
  cam2_stable_local_ids: 4
  cam2_overlapping_duplicate_ids: 2
repro:
  commit: 4211b108ae8e337bc7ecd3fa545117f07a17c0bd
  branch: codex/clip-component-store
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/clip_components_meiji_tracklets_20260925
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-scene-component-meiji-tracklets-20260925
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790333848581759890_592050_scene-component-meiji-tracklets-20260925.log
  evaluation: knowledge/runs/run-scene-component-meiji-tracklets-20260925/evaluation.json
  pipeline_config: knowledge/runs/run-scene-component-meiji-tracklets-20260925/pipeline_config.yaml
  side_confirmation: knowledge/runs/run-scene-component-meiji-tracklets-20260925/side_confirmation.json
parents:
- run-scene-component-meiji-firstframe-20260925
relations: []
papers: []
tags:
- declared_components
- real_clip
- qualification
- incomplete_identity
---

## 観測と判断

先行runで5つに分裂して停止したcam0の人物軌跡は、保存済み検出と原動画を照合して原因を絞った。遠方選手のBoT-SORT ID 2→3→5は13/44frame欠落を挟み、位置差が0.198/0.080 box対角、服装のLab中央値差が1.89/6.33だった。他の同時人物との競合はなく、時間・位置・box寸法・服装色の厳格な一意照合で同一local IDへ結合した。結合前のIDと照合値は`person_tracks` v2のartifactに保存した。cam0/1/2の累計local IDは3/2/4で、各camera上限4を守った。

実clipの3camera×1010frameでCourt先頭frame推定、人物検出・tracking・2D pose・Re-ID、明示確認side、人物/ball三角測量、GVHMR、身体配置、scene assemblyを実行した。外部ballは同じ出力schemaへ変換した3camera分をloadし、ballモデル推論は実行していない。sideは現在のCourtと外部observed ballで4候補を比較した確認根拠を保存し、`[false,false,true]`を明示importした。統合sceneをexport後にchecksumつきで読み直し、全componentのload-only再開を確認した。scene metadataのstatusは`ok`だった。

品質上の制限は大きい。ballの3D有効frameは987/1010だが、4つのglobal人物IDの関節3D有効frame数は`[1010,0,26,50]`、rootは`[815,0,25,0]`、SMPLは`[780,0,25,0]`に偏った。Re-IDの類似度図と人物軌跡図は確認用に生成したが、独立した人手対応・3D正解との比較はまだない。さらにcam2の遠方選手はframe 922に包含関係の重複boxでID 2と9が1frame同時に観測され、その後ID 9へ切り替わった。今回の短期欠落リンクは重複区間を許さず、別local IDとして残った。2 IDが同じ人物であることは原画像の拡大確認と同時frameの重複boxで支持されるため、重複観測の一意照合を追加して再実行する。形式的な完走を人物同一性や3D精度の受入とみなさない。

学習runではなくTensorBoard曲線はない。実行の正確な版・設定・side候補・評価JSONはbundleに固定した。
