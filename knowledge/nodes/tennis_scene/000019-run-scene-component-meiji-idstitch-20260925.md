---
id: run-scene-component-meiji-idstitch-20260925
type: run
task: tennis_scene
sequence: 19
recorded_at: '2026-09-25'
title: cam2重複ID結合後のRe-ID誤対応が幾何検査で停止
issue: 915
provider: codex
session: 01a0d81f-1a8d-7462-b0f8-d827f54c6a4a
date: '2026-09-25'
status: failed
config:
  pipeline: declared_components_v1
  clip: video_000/clip_000
  frames: 1010
  cameras: 3
  court_temporal_policy: static_first_frame
  person_tracks_schema: 3
  person_tracking: unique_short_gap_and_nested_duplicate_links
  ball_source: external_annotation_load
  side_source: geometry_confirmed_load
  confirmed_half_turns: [false, false, true]
metrics:
  cam0_stable_local_ids: 3
  cam1_stable_local_ids: 2
  cam2_stable_local_ids: 3
  cam2_far_person_observed_frames_after_link: 1010
  ball_only_side_cost: 0.103418
  ball_only_side_support: 0.95825
  person_reid_geometry_cost: 0.993452
  person_reid_geometry_support: 0.008646
  combined_side_cost: 0.548435
  combined_side_support: 0.483448
  cost_limit: 0.5
  support_minimum: 0.5
repro:
  commit: 2b084442f29027f4d0402e18bd0193a12b2eddd2
  branch: codex/clip-component-store
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/clip_components_meiji_idstitch_20260925
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-scene-component-meiji-idstitch-20260925
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790336409501183218_784442_scene-component-meiji-idstitch-20260925.log
  evaluation: knowledge/runs/run-scene-component-meiji-idstitch-20260925/evaluation.json
  pipeline_config: knowledge/runs/run-scene-component-meiji-idstitch-20260925/pipeline_config.yaml
  side_confirmation: knowledge/runs/run-scene-component-meiji-idstitch-20260925/side_confirmation.json
  reid_raw_cosine: knowledge/runs/run-scene-component-meiji-idstitch-20260925/reid_raw_cosine.json
  reid_similarity_figure: knowledge/runs/run-scene-component-meiji-idstitch-20260925/reid_similarity.png
  reid_ground_distance_figure: knowledge/runs/run-scene-component-meiji-idstitch-20260925/reid_ground_distance.png
parents:
- run-scene-component-meiji-tracklets-20260925
relations: []
papers: []
tags:
- declared_components
- real_clip
- qualification
- reid_failure
---

## 観測と判断

先行runで残ったcam2遠方選手のBoT-SORT ID 2と9は、frame 922で大小2boxが同じ人物を重複検出し、後者へ切り替わっていた。重複観測1frameのbox包含率は1.0、時間範囲の重複は3frame、服装のLab中央値差は17.63だった。一意な候補だけを結ぶ`person_tracks` v3により、cam2の累計local IDは4→3、該当選手の直接観測は1010frameになった。元IDと照合根拠はartifactに保持し、同時frameは古いIDのboxを採用した。3cameraの人物検出・tracking・2D poseと学習済みRe-IDまでは保存した。

ball観測だけを用いた4候補のside確認は、`[false,false,true]`がcost=0.1034、support=0.9583で優位だった。このsideを明示importし、Re-IDとballの両観測をcamera alignmentへ渡したところ、確認済みsideのcost=0.5484、support=0.4834となり、上限0.5・下限0.5をともに満たさず停止した。人物だけのcost=0.9935、support=0.00865で、ballとは別の問題と切り分けられる。Re-ID出力はcam0のlocal `[1,2,4]`をglobal `[0,1,2]`、cam1の`[1,2]`を`[3,0]`、cam2の`[1,2]`を`[3,0]`に対応させた。

保存済みbboxの足元をballで確認したcamera向きのコート平面へ投影すると、モデルが同一global IDにしたcam0:1とcam1:2の同期frame中央値は27.6m離れていた。cam0:1とcam1:1は0.58m、cam0:2とcam1:2は1.01m、cam0:2とcam2:1は0.67mだった。この距離はpinhole・平面近似とbbox足元の診断値であり、独立した本人正解ではないが、現在のRe-ID割当の幾何不整合を明確に示す。違うside候補への暗黙切替や人物IDの手動上書きは行っていない。新しいcamera_alignment、三角測量、GVHMR、scene exportは未生成であり、`scene.json`に残る先行runの下流参照は現入力に対して古い。component reviewでは依存artifactを確認し、古い出力を明示的に除外している。

モデルembeddingの生cosineは閾値0.775で、cam0:1–cam1:1が0.805、誤ったcam0:1–cam1:2が0.881、cam0:2–cam1:2が0.495だった。したがって閾値の単純な変更だけでは、二組の正しい対応を取り戻せない。全行列の[JSON](../../runs/run-scene-component-meiji-idstitch-20260925/reid_raw_cosine.json)、[数値入り類似度図](../../runs/run-scene-component-meiji-idstitch-20260925/reid_similarity.png)、[足元距離図](../../runs/run-scene-component-meiji-idstitch-20260925/reid_ground_distance.png)を保存した。

前runの形式的完走より、誤対応を検出した今回の失敗を品質判断の根拠とする。幾何的に一意な対応候補は見えるが、既存のRe-ID出力を推論せずに書き換えることはしない。次の方針は、失敗を記録してモデル改善へ進むか、独立した幾何対応componentまたは確認済み人物対応artifactを設計するか、目的を分けて決める必要がある。学習runではなくTensorBoard曲線はない。
