---
id: run-scene-component-meiji-target2-20260925
type: run
task: tennis_scene
sequence: 21
recorded_at: '2026-09-26'
title: 確認済み2人対応で全componentとscene exportを完走
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
  ball_source: external_annotation_load
  side_source: geometry_confirmed_load
  person_reid_source: historical_confirmed_load
  non_target_track_policy: explicit_minus_one_after_trace_check
  confirmed_half_turns: [false, false, true]
metrics:
  camera_local_track_counts: [3, 2, 3]
  confirmed_player_ids: [0, 1]
  scene_player_axes: 2
  player_joint_valid_frames: [1010, 995]
  player_root_valid_frames: [1004, 995]
  player_smpl_valid_frames: [1004, 995]
  ball_3d_valid_frames: 987
  load_only_stage_count: 25
repro:
  commit: 96d6cd90b38dc049a7d2ef27d27628eefb5c7617
  branch: codex/clip-component-store
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/clip_components_meiji_target2_20260925
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-scene-component-meiji-target2-20260925
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790347180070683871_1243141_scene-component-meiji-target2-20260925.log
  evaluation: knowledge/runs/run-scene-component-meiji-target2-20260925/evaluation.json
  pipeline_config: knowledge/runs/run-scene-component-meiji-target2-20260925/pipeline_config.yaml
  side_confirmation: knowledge/runs/run-scene-component-meiji-target2-20260925/side_confirmation.json
  person_confirmation: knowledge/runs/run-scene-component-meiji-target2-20260925/person_confirmation.json
  reid_raw_cosine: knowledge/runs/run-scene-component-meiji-target2-20260925/reid_raw_cosine.json
  reid_similarity_figure: knowledge/runs/run-scene-component-meiji-target2-20260925/reid_similarity.png
  reid_ground_distance_figure: knowledge/runs/run-scene-component-meiji-target2-20260925/reid_ground_distance.png
  scene_figure: knowledge/runs/run-scene-component-meiji-target2-20260925/scene_topdown.png
  body_figure: knowledge/runs/run-scene-component-meiji-target2-20260925/body_mesh_samples.png
parents:
- run-scene-component-meiji-confirmed-reid-20260925
relations: []
papers: []
tags:
- declared_components
- real_clip
- qualification
- confirmed_target_players
---

## 観測と判断

ユーザーが確認した対象プレーヤー2人とraw人物trackを区別した。cam0の追加track ID4は現対象の旧bbox軌跡との最近接中央値が2.61 box寸法で、確認済み2人のいずれでもない。cam2の1frame track ID5はモデル有効poseがない。両者はraw検出・tracking・pose成果物に残すが、確認済み`person_identities`ではID `-1`とし、下流の人物対応・三角測量・GVHMR・scene player軸から明示的に除外した。旧対象に近い有効extra trackなら停止する検査を加え、静かな棄却を防いだ。学習済みRe-IDの生embedding・推論IDは別のimmutable artifactとして保持し、確認済みIDをモデル推論の成果とは扱わない。

3camera×1010frameでCourt frame 0のKP＋LINE共同推定、人物検出・tracking・2D pose、外部ball/確認済みside/確認済み人物対応load、camera alignment、人物/ball三角測量、GVHMR、身体配置、scene assemblyを完走した。scene player IDは`[0,1]`のみで、関節3D有効frameは`[1010,995]`、rootとSMPL配置はともに`[1004,995]`、ball 3Dは987frame。scene exportをchecksumつきで再読込し、全25componentのload-only再開を確認した。上流の採用artifactが変わると旧export公開参照を外し、scene readerも依存鎖を検証するため、途中の古いsceneを現行結果として返さない。

これは確認済み人物対応を使った下流の構造・処理完走の証拠であり、学習済みRe-IDが実clipで正しいという評価ではない。[前run](000019-run-scene-component-meiji-idstitch-20260925.md)の生cosineと幾何診断が示すRe-ID誤対応は残る。独立した人手3D正解、他clip/会場での対象選別・3D精度は未確認。可視化での目視評価とモデル側の改善を別に進める。学習runではなくTensorBoard曲線はない。

全componentの目視用画像・timeline・12本のH.264全frame overlay動画は、clip store外の`outputs/tennis_scene/evaluate/clip_components_meiji_target2_20260925/component_review/`に生成した。PR内で確認できる最小の固定証拠として[2人のscene俯瞰図](../../runs/run-scene-component-meiji-target2-20260925/scene_topdown.png)、[身体頂点のサンプル図](../../runs/run-scene-component-meiji-target2-20260925/body_mesh_samples.png)、[モデル生cosine JSON](../../runs/run-scene-component-meiji-target2-20260925/reid_raw_cosine.json)をbundleに収めた。
