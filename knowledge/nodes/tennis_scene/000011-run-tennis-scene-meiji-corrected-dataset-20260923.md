---
id: run-tennis-scene-meiji-corrected-dataset-20260923
type: run
task: tennis_scene
sequence: 11
recorded_at: '2026-09-23'
title: 'Meiji全区間: 正規化修正後のdataset scene・7動画と出力評価'
provider: codex
session: 01a0c8c3-d190-7c51-b671-ded223340a2f
date: '2026-09-23'
status: done
config:
  clip: video_000/clip_000
  frames: 1010
  fps: 59.94006
  cameras:
  - cam0
  - cam1
  - cam2
  reference_camera: cam0
  view_half_turns:
  - false
  - false
  - true
  ball_normalize_imagenet: true
  court_source: own task-fresh independently regenerated result, explicitly approved
    reuse
  human_source: independent DINO/ViTPose/HMR2/GVHMR execution
  scale_mode: fixed
metrics:
  finite_arrays: 22
  movies: 7
  movie_frames: 1010
  court_mean_px:
    cam0: 9.927464882672615
    cam1: 13.129072637822086
    cam2: 8.374166851061755
  ball_missing_rate:
    cam0: 0.6254458977407849
    cam1: 0.3390862944162436
    cam2: 0.3400637619553666
  ball_median_px:
    cam0: 7.635747936174171
    cam1: 4.2954947491485935
    cam2: 6.620660786347368
  ball_p95_px:
    cam0: 475.8777984685381
    cam1: 311.5866299183623
    cam2: 336.7002860406026
  ball_negative_height_frames: 50
  ball_max_speed_m_s: 412.1759948730469
  player_max_speed_m_s: 36.56314468383789
  alignment_position_rmse_m:
  - 1.6927050583916503
  - 2.213240734962118
  slcs_reader:
    accepted: true
    player_valid_frames:
    - 1010
    - 1010
    ball_valid_frames: 978
repro:
  commit: fa8798a4c19f7efc6b123f4819130d57fe14b565
  branch: codex/tennis-scene-responsibility-cleanup
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg /home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup/.venv/bin/python
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/checked_entrypoint.py
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset.json
artifacts:
  run_dir: knowledge/runs/run-tennis-scene-meiji-corrected-dataset-20260923
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790129435511577647_3356974_tennis-scene-cleanup-20260923T020139Z-dataset.log
parents:
- run-ball-checkpoint-normalization-meiji-20260923
- run-tennis-scene-meiji-raw-ball-baseline-20260923
relations:
- to: run-tennis-scene-meiji-corrected-pipeline-20260923
  rel: compares
papers: []
tags:
- real_clip
- scene_publication
- model_only
- quality_limits
---

## 結果と実行範囲

指定clipの全1010frame・3cameraでscene生成を完了した。ユーザー承認に従い、Courtは各入口がこのタスク内で独立生成した完了結果を引き継いだ。単発側の人物推定はこのタスク内の再生成・映像確認済み結果を追加承認で引き継ぎ、dataset側は人物推定を独立に実行した。PLCS・Ball・BLCS・motion alignmentとscene/可視化は修正版で再実行した。元から存在した旧scene・旧自動予測は生成入力ではない。

人物対応は全frameのbbox/2D姿勢、通常時点と低margin時点の元映像・cropを確認した。両入口とも[[0,0,1],[1,1,0]]、逆順が優位なframeは0。微小なpose差はあるが、対象を入れ替えた証拠はない。7本の動画は全frameデコードし、各1010frame・約59.94fps・約16.850秒・blank判定0。通常/極端値frameの描画を確認した。sceneの22配列、metadata、軸と完成マーカーの整合を確認した（単発にはdataset完成マーカーを付けない）。

## 観測整合と残る品質問題

Courtのcamera-local手動点距離平均は約9.93/13.13/8.37px、全camera全frameで14点が成立した。静的手動点に対する同一clip内の反復比較であり、独立な3D精度評価ではない。

Ballのobserved frame欠損率は約62.5/33.9/34.0%、検出時中央値距離は7.64/4.30/6.62px。大きな誤検出も残り、p95は約476/312/337px。保存RGB正規化を復元して欠損と平均距離は改善したが、すべての指標が改善したわけではない。

3D ballは50frameでz<0、最大frame間速度412.18m/s。f807→808ではcam2の2D ballが約337px移り、原映像でも別位置への移動が見える。PLCSにも約36.56m/sのjumpがあり、cam0の13関節mask復帰や他viewの姿勢変動が同時にある。両例は時間窓の境界ではなく、結合・軸・FPS・二重denormalizeの不整合を示す証拠はなかった。個別窓の未blend予測は未保存で、厳密な寄与分離は未確認。

GVHMR整列ソルバは成功したがPLCSとの位置RMSEは約1.69/2.22m、heading差と高さ変動も残る。3D実測GTはなく、これらを3D精度と呼ばない。生成sceneを高品質教師GTとみなせない。

## 両生成入口と証拠

最大要素差はPLCS位置0.01241m、整列GVHMR位置0.02471m、ball3D 0.00006098m。yawは±πを考慮した円周差でPLCS最大0.331°、整列GVHMR最大0.586°。設定差は出力/引継ぎパス・GVHMR source・保存共通configのvideo_pathsで、実metadataの動画/cameraは一致した。全配列のbit一致ではない。

evidenceに設定・由来・SHA・人物照合・全指標・確認画像を保存した。scene/動画の実体は大きいため外部artifactとして保持し、evaluation.jsonに絶対パスとSHAを記録した。学習ではなくTensorBoard曲線はない。手動Courtと外注Ballは評価のみに使用した。
