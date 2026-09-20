---
task: tennis_scene
sequence: 1
recorded_at: 2026-09-13
date_source: experiment_date
papers: []
id: run-tennis-scene-meiji-clip000-plcs-e19
type: run
title: Meiji 3-camera clip_000 PLCS epoch 19 inference
provider: codex
date: '2026-09-13'
status: done
config:
  model: ckpt/plcs/plcs-multiview-axial-split-camera-view-v2-epoch19.ckpt
  loss: inference_only
  data: data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
metrics:
  cameras: 3
  players: 2
  source_frames: 1010
  source_fps: 59.94
  plcs_sample_stride: 2
artifacts:
  log: .training_queue/logs/1789299324178750358_3450711_tennis_scene_meiji_clip000_plcs_epoch19_final2.log
  output_dir: data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/tennis_scene
parents: []
relations: []
tags:
- tennis_scene
- plcs
- camera_view_v2
- gvhmr
- real_data
---

## 考察 / Findings

- `camera_view_v2` として `cam0` をreference camera、view half-turnを
  `[false, false, true]` に固定し、手動CourtKP14を全カメラで同じreference frameへ
  permutationしてからPLCSへ渡した。typed reference metadataも2 player分を明示した。
- 無制約のDINO + BoT-SORTで面積上位2 trackを選ぶと、cam0/cam2では隣接コートの
  人物が奥側playerより大きく写り、対象playerを置き換えた。CourtKP14から対象コートを
  画像へ射影し、sideline 1 m・baseline 5 mのmarginを持つfootpoint polygonで検出を
  絞ると、cam0/cam1/cam2の全てでtrack 1が手前、track 2が奥として1010 frameを維持した。
  frame 1/300/700/900を各cameraで目視確認し、単一区間のidentity associationを保存した。
- 59.94 fps入力を`sample_stride=2`で約29.97 fpsへ落として7個の128-frame windowで推論し、
  positionを線形補間、yawをheading vector補間して1010 source frameへ復元した。
- 公開されたplayer position/yawは全要素finite。PLCS-only実行ではball stagesを無効化し、
  stage-aware dataset contractによりball配列を要求せずsceneを生成できた。
