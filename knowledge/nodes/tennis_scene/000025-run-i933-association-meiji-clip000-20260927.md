---
id: run-i933-association-meiji-clip000-20260927
type: run
task: tennis_scene
sequence: 25
recorded_at: '2026-09-27'
title: 人物対応をcomponentで実行したimportなし（ball注釈を除く）Meiji clip_000実clip qualification
issue: 933
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
  side_source: execute court_side
  player_association_source: execute player_association (geometry + CLIP-ReID appearance, MILP)
  association_config: src/tasks/player_association/configs/association.yaml
  players_per_side: 1
  max_tracks_per_camera: 16
  association_labels: tests/benchmarks/labels/player_association/meiji_3cam/video_000/clip_000.json
metrics:
  association_pair_f1: 1.0
  association_pair_tp: 5887
  association_group_accuracy: 1.0
  association_exclusion_precision: 1.0
  association_exclusion_recall: 1.0
  association_id_switch_true: 0
  association_id_switch_predicted: 0
  association_min_player_margin: 8.04
  association_candidates: 7
  association_undecided_candidates: 0
  scene_player_axes: 2
  player_joint_valid_frames: [1010, 995]
  player_root_valid_frames: [1004, 995]
  player_smpl_valid_frames: [1004, 995]
  ball_3d_valid_frames: 987
  load_only_resume_nodes: 25
  scene_npz_max_abs_diff_vs_i932: 0.0
repro:
  commit: ed12854b89fd8896552bd1577098f6361a835128
  branch: campaign930/i933-4-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: bash tests/benchmarks/build_dino_extension.sh /home/kamimura/projects/tennis-lab
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i933-association-meiji-clip000-20260927/dino_extension
    && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i933-association-meiji-clip000-20260927/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/i933-association-meiji-clip000-20260927
    --association-labels /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i933-player-association/tests/benchmarks/labels/player_association/meiji_3cam/video_000/clip_000.json
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-i933-association-meiji-clip000-20260927
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790514115733425930_3028061_i933-association-meiji-clip000-20260927.log
  evaluation: knowledge/runs/run-i933-association-meiji-clip000-20260927/evaluation.json
  pipeline_config: knowledge/runs/run-i933-association-meiji-clip000-20260927/pipeline_config.yaml
  player_association: knowledge/runs/run-i933-association-meiji-clip000-20260927/player_association.json
  output_dir: outputs/tennis_scene/evaluate/i933-association-meiji-clip000-20260927
parents:
- run-i932-component-side-meiji-clip000-20260927
relations:
- to: run-i932-component-side-meiji-clip000-20260927
  rel: compares
- to: run-i933-association-meiji
  rel: compares
papers: []
tags:
- declared_components
- qualification
- player_association
---

## 要約

#933 の PR #946 の tree（`ed12854b`）で、[run-i932-component-side-meiji-clip000-20260927](000024-run-i932-component-side-meiji-clip000-20260927.md) と同じ benchmark を Meiji `clip_000`（3 camera × 1010 frame）に再実行した。
違いは人物対応の出どころだけである。000024 までは過去に人手で確認した対応を import していたが、今回は既定の `pipeline.yaml` で `player_association: execute` とし、`player_association` component が track・コート較正・`court_side` の出力と動画の crop から対応を決めた（幾何＋CLIP-ReID 外観、MILP、[run-i933-association-meiji](../player_association/000002-run-i933-association-meiji.md) と同じ `association.yaml`）。
import は ball（外注 `video_ball_annotation.v2` の observed 点）だけになった。

結果は status=ok。benchmark に `--association-labels` で評価ラベル（#944）を渡し、component の出力（`person_identities` v3、frame ごとの ID）を box 単位で照合した。

| 指標 | 値 |
|---|---|
| pair F1（P / R） | 1.000（1.000 / 1.000、tp 5887） |
| group accuracy | 1.000（1010 / 1010 frame） |
| 除外 P / R | 1.000 / 1.000（隣コートの人 cam0 t4、179 box） |
| ID switch（正解 / 予測） | 0 / 0 |
| 選手に関わる決定の最小マージン | 8.04 |
| 停止・未決定の区間 | なし |

label box は 3 camera の 6146 個すべてが予測 box と対応した（`coverage`）。cam2 t5 は A の重複 box（f139 の 1 frame だけの track）で、候補から外れて `-1` になった。同じ frame の A は t2 が ID を持つので、指標の定義上は誤りではない（重複 box の `-1` は除外の誤りに数えない）。

## 既存実験との比較

000024（人物対応を import）と比べて、有効 frame 数（関節 3D [1010, 995]、root/SMPL [1004, 995]、ball 3D 987）が一致した。
出力した `scene.npz` の全 29 配列を比べ、**最大絶対差 0.0、NaN の位置も一致**した。人物対応を import から component に替えても、player の番号（side -1 = player 0）を含め scene は bit 単位で同じである。
全 25 node の load-only 再開も完走した。

`pipeline_config.yaml` の差は、`execution.player_association` の `load → execute`、`player_association.config`・`players_per_side: 1` の追加、`person_tracking.max_tracks_per_camera: 16`（#944 の判断）である。clip_000 の track 数は camera あたり 2〜3 個なので、上限の変更は tracking の結果に影響しない。

## 解釈と限界

- 確かめたのは、#933 の方式が pipeline の component として import なしに動き、人手で確認した対応と完全一致すること。clip_000 は評価 4 clip のうち最も易しい（シングルス、ID switch なし）。難しい 3 clip の指標は [run-i933-association-meiji](../player_association/000002-run-i933-association-meiji.md) にある（pipeline と同じ関数で評価した）。
- ball は外注注釈のまま（#934/#935 で置き換える）。side もこの ball から決めている。
- 3D 精度は独立した基準と比べていない（000022・000024 と同じ）。

## 次に有効な実験

#934/#935 で ball の import が外れたら、同じ benchmark を import なしで再実行する。
ダブルスの clip は評価ラベルに無いので、ダブルスの Meiji clip が得られたら `players_per_side: 2` でラベルを追加して評価する。
