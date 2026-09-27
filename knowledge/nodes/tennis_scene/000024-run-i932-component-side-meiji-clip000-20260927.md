---
id: run-i932-component-side-meiji-clip000-20260927
type: run
task: tennis_scene
sequence: 24
recorded_at: '2026-09-27'
title: court_sideをcomponentで決めたMeiji clip_000実clip qualification
issue: 932
provider: claude
session: abf284fe-984d-4b6c-afec-7375982b0bb0
date: '2026-09-27'
status: done
config:
  pipeline: declared_components_v1
  config_source: src/tennis_scene/configs/pipeline.yaml (default)
  overrides: [paths.*, device=cuda, output_directory=run, execution.ball_detection=load]
  clip: video_000/clip_000
  frames: 1010
  cameras: 3
  ball_source: import video_ball_annotation_v2
  side_source: execute court_side (ball-only half-turn hypothesis test)
  court_side_thresholds: {reprojection_px: 20.0, min_motion_px: 5.0, min_frames: 8, max_cost: 0.8, min_support: 0.2, min_margin: 0.15}
  player_association_source: import confirmed_historical_person_association
metrics:
  court_side_view_half_turns: [false, false, true]
  court_side_evidence_frames: 493
  court_side_best_cost: 0.102
  court_side_best_support: 0.961
  court_side_next_cost: 0.707
  court_side_margin: 0.605
  scene_player_axes: 2
  player_joint_valid_frames: [1010, 995]
  player_root_valid_frames: [1004, 995]
  player_smpl_valid_frames: [1004, 995]
  ball_3d_valid_frames: 987
  load_only_resume_nodes: 25
  scene_npz_max_abs_diff_vs_i931: 0.0
repro:
  commit: 6f8c44fc34be6983e993d6cc6871a312872c62c8
  branch: campaign930/i932-2-component
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: bash tests/benchmarks/build_dino_extension.sh /home/kamimura/projects/tennis-lab
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i932-component-side-meiji-clip000-20260927/dino_extension
    && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i932-component-side-meiji-clip000-20260927/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i932-component-side-meiji-clip000-20260927
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-i932-component-side-meiji-clip000-20260927
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790479641035680991_3379220_i932-component-side-meiji-clip000-20260927.log
  evaluation: knowledge/runs/run-i932-component-side-meiji-clip000-20260927/evaluation.json
  pipeline_config: knowledge/runs/run-i932-component-side-meiji-clip000-20260927/pipeline_config.yaml
  person_confirmation: knowledge/runs/run-i932-component-side-meiji-clip000-20260927/person_confirmation.json
  output_dir: outputs/tennis_scene/evaluate/i932-component-side-meiji-clip000-20260927
parents:
- run-i931-default-meiji-clip000-20260927
relations:
- to: run-i931-default-meiji-clip000-20260927
  rel: compares
- to: run-i932-meiji-clips-side-20260927
  rel: compares
papers: []
tags:
- declared_components
- qualification
- court_side
---

## 要約

#932 の PR #942 の tree（`6f8c44fc`）で、#931 の [run-i931-default-meiji-clip000-20260927](000022-run-i931-default-meiji-clip000-20260927.md) と同じコマンドを Meiji `clip_000`（3 camera × 1010 frame）に再実行した。
違いは court side の決め方だけである。#931 では side を import（ball で確認した結果を読み込む）していたが、今回は既定の `pipeline.yaml` で `court_side: execute` とし、`court_side` component が ball だけの half-turn 仮説検定で side を決めた。
import は ball（外注 `video_ball_annotation.v2` の observed 点）と player association の2つだけになった。

結果は status=ok。`court_side` は `[False, False, True]` を選んだ。証拠 frame は 493、最良の仮説の cost 0.102・support 0.961、次点（`[False, True, False]`）の cost 0.707 で、margin は 0.605（閾値 0.15）。
`camera_alignment` の再検証も通り、全 25 node の load-only 再開も完走した。

## 既存実験との比較

000022 と比べて、有効 frame 数（関節 3D [1010, 995]、root/SMPL [1004, 995]、ball 3D 987）と人物対応の照合結果（`person_confirmation.json`）が一致した。
さらに、出力した `scene.npz` の全配列（ball 3D、player 位置・yaw・関節、SMPL パラメータと頂点など 28 key）を比べ、**最大絶対差 0.0、NaN の位置も一致**した。side の決め方を import から component に替えても、scene は bit 単位で同じである。

side の cost が 000022 の 0.103 から 0.102 に、評価した frame 数が 506 から 493 に変わった。
これは `court_side` が静止した重複観測を証拠から除く（`min_motion_px` 5 px@1080p、[run-i932-synthetic-side-thresholds](../court_side/000001-run-i932-synthetic-side-thresholds.md)）ためで、判定と下流には影響しない。

## 解釈と限界

- ball は外注注釈なので、これは「注釈 ball のもとで component 化した side が既定設定で動く」ことの確認である。検出器 ball での side の判定率（Meiji 51 clip 中 7 clip）は [run-i932-meiji-clips-side-20260927](../court_side/000002-run-i932-meiji-clips-side-20260927.md) を見ること。
- 3D 精度は独立した基準と比べていない（000022 と同じ）。

## 次に有効な実験

#933（人物対応）と #934/#935（ball）のモデルが入ったら、残る import を外して同じ benchmark を再実行する。
検出器 ball だけで side が決まらない clip では、その時点で `court_side_<理由>` で停止するはずなので、停止理由も合わせて記録する。
