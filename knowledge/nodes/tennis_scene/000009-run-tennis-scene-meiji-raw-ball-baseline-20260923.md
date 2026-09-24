---
id: run-tennis-scene-meiji-raw-ball-baseline-20260923
type: run
task: tennis_scene
sequence: 9
recorded_at: '2026-09-23'
title: 'Meiji全区間sceneと7動画: Ball正規化欠落を含む診断baseline'
provider: codex
session: 01a0c8c3-d190-7c51-b671-ded223340a2f
date: '2026-09-23'
status: done
config:
  clip: video_000/clip_000
  frames: 1010
  cameras:
  - cam0
  - cam1
  - cam2
  court_source: this-task regenerated Court, reused with explicit user approval
  gvhmr_court_filter_baseline_margin_m: 10.0
  ball_normalize_imagenet: false
metrics:
  finite_scene_arrays: 22
  decoded_movies: 7
  decoded_frames_per_movie: 1010
  ball_missing_on_observed_rate:
  - 0.8406658739595719
  - 0.35126903553299493
  - 0.44845908607863977
  ball_height_below_zero_frames: 301
  ball_max_interframe_speed_m_s: 310.116455078125
  plcs_max_interframe_speed_m_s: 36.56344223022461
  alignment_position_rmse_m:
  - 1.6913527542332083
  - 2.2173260677136035
repro:
  commit: fa8798a4c19f7efc6b123f4819130d57fe14b565
  branch: codex/tennis-scene-responsibility-cleanup
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg /home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup/.venv/bin/python
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T231809Z/checked_entrypoint.py
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T231809Z/pipeline.json
artifacts:
  run_dir: knowledge/runs/run-tennis-scene-meiji-raw-ball-baseline-20260923
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790119155380520984_2342058_tennis-scene-cleanup-20260922T231809Z-pipeline.log
parents:
- run-tennis-scene-meiji-visible-court-roi-20260922
relations:
- to: run-ball-checkpoint-normalization-meiji-20260923
  rel: compares
papers: []
tags:
- real_clip
- diagnostic
- preprocessing_mismatch
---

## 観測結果

承認されたこのタスク内の新規Courtを引き継ぎ、3camera全1010frameのDINO・ViTPose・HMR2・GVHMR、PLCS、Ball、BLCS、motion alignmentを実行した。元からあった推論結果・sceneは生成入力にしていない。人手の選手対応は全frameの新旧bbox・2D姿勢と通常/低margin時点の元映像・人物cropを照合してから適用した。対応は [[0,0,1],[1,1,0]]、逆順が優位なframeは全cameraで0。cam2遠方選手の終盤など、人物同一性が確認できても姿勢点がずれる場面は残る。

sceneの22配列は有限で、1010frame・3camera・2選手の軸は成立した。visualizationと全6タスクのvisualize_tasksは終了コード0。7本すべてを全frameデコードし、各1010frame、約59.94fps、約16.850秒、blank判定0を確認した。これは品質の合格ではない。

Ballのobserved frame欠損率はcam0/1/2で84.1/35.1/44.8%。3D ballは301frameでz<0、最大frame間速度310.1m/s。PLCSにも36.6m/sの不連続候補がある。GVHMR整列のソルバは両選手で成功したが、PLCSとの位置RMSEは1.69/2.22m、heading残差も残り、真の3D精度とはみなせない。

## 原因と取り扱い

公開Ball APIのRGB[0,1]制約を通すために設定したnormalize_imagenet=falseが、checkpoint保存のenabled=trueを再現していなかった。MDDに入る前のchannel別stdが異なるため同値ではなく、重みの品質評価と前処理バグが交絡している。sceneと動画は診断baselineとして保存し、最終の有効な成果物とは扱わない。修正後の全区間再評価が必要である。3D実測GTはない。

このrunは学習ではなくTensorBoard曲線はない。コマンド・source差分はqueue bundle、入力識別子・設定・人物照合・動画decode結果・参考図はevidenceに保存した。大きなscene・動画自体の絶対パスとSHAはevaluation-pipeline.jsonにある。
