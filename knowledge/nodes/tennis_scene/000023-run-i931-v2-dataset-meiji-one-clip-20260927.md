---
id: run-i931-v2-dataset-meiji-one-clip-20260927
type: run
task: tennis_scene
sequence: 23
recorded_at: '2026-09-27'
title: v1 layout廃止後のSLCS dataset再生成（Meiji clip_000、v2 SceneResult公開）
issue: 931
provider: claude
session: abf284fe-984d-4b6c-afec-7375982b0bb0
date: '2026-09-27'
status: failed
config:
  pipeline: declared_components_v1
  config_source: src/tennis_scene/configs/pipeline.yaml (default)
  mode: tests/benchmarks/component_pipeline.py --dataset --seed-from
  source_clip: data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
  dataset: data/slcs/meiji_one_clip_scene_v2
  replaces_v1_dataset: data/slcs/meiji_one_clip_repro_takeover001_a
  clip: video_000/clip_000
  frames: 1010
  cameras: 3
  ball_source: import video_ball_annotation_v2
  side_source: import ball_confirmed_court_side
  player_association_source: import confirmed_historical_person_association
metrics:
  pipeline_status: ok
  confirmed_half_turns: [false, false, true]
  player_joint_valid_frames: [1010, 995]
  player_root_valid_frames: [1004, 995]
  player_smpl_valid_frames: [1004, 995]
  ball_3d_valid_frames: 987
  slcs_reader_num_frames: 1010
  slcs_reader_schema_version: 2
  plcs_residual_calibration_frame_index: 0
  plcs_residual_player_ids: [0, 1]
repro:
  commit: f3706f1913bee277ea950be4252af0b3cb12a389
  branch: campaign930/i931-4-imports
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: bash tests/benchmarks/build_dino_extension.sh /home/kamimura/projects/tennis-lab
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i931-v2-dataset-meiji-one-clip-20260927/dino_extension
    && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i931-v2-dataset-meiji-one-clip-20260927/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --dataset /home/kamimura/projects/tennis-lab/data/slcs/meiji_one_clip_scene_v2
    --seed-from /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --clip /home/kamimura/projects/tennis-lab/data/slcs/meiji_one_clip_scene_v2/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i931-v2-dataset-meiji-one-clip-20260927
    --device cuda && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.tasks.slcs.scripts.precompute_dino_tokens
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i931-v2-dataset-meiji-one-clip-20260927/slcs
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/third_party/dinov3/checkpoints
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party precompute.checkpoint_path=dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
    data.dataset_root=slcs/meiji_one_clip_scene_v2
artifacts:
  run_dir: knowledge/runs/run-i931-v2-dataset-meiji-one-clip-20260927
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790468591832132284_2084456_i931-v2-dataset-meiji-one-clip-20260927.log
  evaluation: knowledge/runs/run-i931-v2-dataset-meiji-one-clip-20260927/evaluation.json
  pipeline_config: knowledge/runs/run-i931-v2-dataset-meiji-one-clip-20260927/pipeline_config.yaml
  side_confirmation: knowledge/runs/run-i931-v2-dataset-meiji-one-clip-20260927/side_confirmation.json
  person_confirmation: knowledge/runs/run-i931-v2-dataset-meiji-one-clip-20260927/person_confirmation.json
  output_dir: outputs/tennis_scene/evaluate/i931-v2-dataset-meiji-one-clip-20260927
  dataset_dir: data/slcs/meiji_one_clip_scene_v2
parents:
- run-i931-default-meiji-clip000-20260927
relations:
- to: run-i931-default-meiji-clip000-20260927
  rel: confirms
papers: []
tags:
- declared_components
- dataset_regeneration
- scene_result_v2
---

## 要約

ユーザー判断（#931、v1 annotation layout の廃止）に従い、v1 layout の SLCS dataset `data/slcs/meiji_one_clip_repro_takeover001_a`（clip_000 のみ）を、**新しい出力先** `data/slcs/meiji_one_clip_scene_v2` に v2 SceneResult として再生成した。旧 dataset は削除していない。

queue job は 2 段構成で、**pipeline 段は成功し、後段の DINO precompute 段が設定エラーで失敗した**（job 全体の status は failed）。

- pipeline 段（`component_pipeline.py --dataset --seed-from`）: status=ok。source clip から clip を作り（media は hard link、import 入力は copy）、store を `<clip>/annotations/tennis_scene` に書き、本番の `generate_pseudo_annotations` で `annotation.json` を公開した。その後 SLCS reader（1010 frame、schema_version=2）と PLCS residual reader（校正 frame 0、player [0, 1]）で読み戻せた。
- precompute 段: `paths.output_root=<report>/slcs` と指定したため、ログ出力先 `slcs/precompute/...` の先頭 `slcs` が「設定した root の basename」と衝突し、path contract（`src/utils/configuration/paths.py`）が停止した。contract の挙動は意図どおりで、原因は起動時の指定。ただしエラー文が衝突した root を示さず、原因を特定しにくかったため、4f1001dc でエラー文に root 名を含めた。precompute は別 job [run-i931-v2-dataset-meiji-dino-precompute-20260927](../slcs/000139-run-i931-v2-dataset-meiji-dino-precompute-20260927.md) でやり直した。

## 既存実験との比較

各 validity は [run-i931-default-meiji-clip000-20260927](000022-run-i931-default-meiji-clip000-20260927.md)（同じ既定設定と import、store は report 配下）と一致した。関節 3D は [1010, 995]、root/SMPL は [1004, 995]、ball 3D は 987、half-turn は `[False, False, True]`。store の置き場所を clip の `annotations/` に変えても、結果は変わらないことを確認した。

## 限界と次の実験

- 観測: 再生成できたのは、手動人物対応がある Meiji `video_000/clip_000` だけ。残りの v1 dataset（`data/` 配下 23 root、218 clip）は、`player_association` と `court_side` を推論で作れるようになる（#932/#933、M1）まで同じ経路で再生成できない。broadcast 単眼の dataset は v2 に 3D を持たないため、再生成の対象外とした（#931 の【要判断】）。
- split file は作っていない。学習に使う場合は `make_splits` で作る。
