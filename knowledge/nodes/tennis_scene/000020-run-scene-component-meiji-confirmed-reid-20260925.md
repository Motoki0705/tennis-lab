---
id: run-scene-component-meiji-confirmed-reid-20260925
type: run
task: tennis_scene
sequence: 20
recorded_at: '2026-09-25'
title: 確認済み対応で完走したが非対象trackが3人目へ流入
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
  ball_source: external_annotation_load
  side_source: geometry_confirmed_load
  person_reid_source: historical_confirmed_load
  confirmed_target_players: 2
  extra_track_policy: model_valid_singleton
metrics:
  scene_player_axes: 3
  player_joint_valid_frames: [1010, 995, 0]
  player_root_valid_frames: [1004, 995, 0]
  player_smpl_valid_frames: [1004, 995, 0]
  extra_axis_gvhmr_observed_samples: 90
  ball_3d_valid_frames: 987
repro:
  commit: d236a7184651d354a8c684eb45e4324868bc48c9
  branch: codex/clip-component-store
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/clip_components_meiji_confirmed_reid_20260925
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-scene-component-meiji-confirmed-reid-20260925
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790342273367562285_977574_scene-component-meiji-confirmed-reid-20260925.log
  evaluation: knowledge/runs/run-scene-component-meiji-confirmed-reid-20260925/evaluation.json
  pipeline_config: knowledge/runs/run-scene-component-meiji-confirmed-reid-20260925/pipeline_config.yaml
  side_confirmation: knowledge/runs/run-scene-component-meiji-confirmed-reid-20260925/side_confirmation.json
  person_confirmation: knowledge/runs/run-scene-component-meiji-confirmed-reid-20260925/person_confirmation.json
parents:
- run-scene-component-meiji-idstitch-20260925
relations: []
papers: []
tags:
- declared_components
- real_clip
- qualification
- extra_player_axis
---

## 観測と判断

学習済みRe-IDの生embedding・予測IDをimmutable artifactに保存した後、既存の`player_association_result.json`が指す旧GVHMR player軸を、各cameraのbbox時系列で現trackへ一意に照合した。対象2名の中心差中央値は各軸0で、最良以外との距離差も検査した。確認済み対応を同じ`person_identities`出力schemaで明示importし、model artifactのembedding・valid mask・cosine閾値は変更していない。ballとsideも以前どおり明示loadした。

この確認済み対応によりcamera alignmentのcost=0.1174、support=0.9685で幾何検査を通過し、全clipの三角測量・GVHMR・scene exportとload-only再開は完走した。ball 3Dは987/1010frame、対象2名の関節3Dは1010/995frame、SMPL配置は1004/995frame有効だった。ただし検証用importがcam0の対象外人物track ID4を単独global ID2として残し、統合sceneのplayer軸が`[0,1,2]`になった。3人目は関節3D・root・SMPL有効frameが0でも、body view selectionとGVHMRに90観測サンプルが渡っていた。ユーザーが確認した「このclipの対象プレーヤーは2人」という条件に反するため、形式的な完走を受入結果とはしない。

次版ではraw人物検出・tracking・poseの3本目を消さず、確認済み対象2名に含まれないtrackをID `-1`として明示的に除外し、三角測量・GVHMR・sceneのplayer軸を2人に固定する。cam2の1frame余剰trackはモデル有効poseなしとして引き続き除外する。bbox軌跡が旧対象へ近い追加trackを静かに除外しない検査も追加する。Re-IDモデル自体の実clip精度は依然として不合格であり、この確認済み対応による下流成功をモデル精度へ転用しない。

学習runではなくTensorBoard曲線はない。実行時の正確な設定・確認済み対応・評価はbundleを参照する。
