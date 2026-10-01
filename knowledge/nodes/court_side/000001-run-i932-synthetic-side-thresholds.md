---
id: run-i932-synthetic-side-thresholds
type: run
task: court_side
sequence: 1
recorded_at: '2026-09-27'
title: 'court_side: 合成ベンチマークによるballだけのside判定の閾値選定'
issue: 932
provider: claude
date: '2026-09-27'
status: done
config:
  dataset: blcs/single_object_camera_view_v2 (test split, 30 fps, 4 physical cameras, 1280x720)
  selection: seed 0, test scenes 0-599, 28 conditions x 600 trials
  holdout: seed 1, test scenes 600-999, 28 conditions x 400 trials, thresholds fixed from selection
  scoring: reprojection_px 20 and min_motion_px 5 at 1920x1080, scaled by image diagonal
  grid: max_cost 0.3-0.9, min_support 0.1-0.6, min_margin 0.02-0.3, min_frames 4-60 (1260 points)
  rule: lowest mean stop rate among points that, with every one-step looser neighbour, make no wrong decision
  code: campaign930/i932-1-core b66db664
metrics:
  selected_min_frames: 8
  selected_max_cost: 0.8
  selected_min_support: 0.2
  selected_min_margin: 0.15
  selected_min_motion_px: 5.0
  holdout_wrong_decisions: 0
  holdout_stop_rate_nominal: 0.0
  holdout_stop_rate_combined: 0.145
  holdout_previous_stop_rate_combined: 0.595
  largest_wrong_argmin_margin_selection: 0.103
  largest_wrong_argmin_margin_holdout: 0.088
artifacts:
  run_dir: knowledge/runs/run-i932-synthetic-side-thresholds
  output_dir: outputs/court_side/evaluate/synthetic_blcs_v2/i932-select-s0-600-20260927
  holdout_output_dir: outputs/court_side/evaluate/synthetic_blcs_v2/i932-holdout-s1-400-20260927
  command: .venv/bin/python -m src.tasks.court_side.scripts.benchmark_synthetic --data-root <repo>/data --output-root <repo>/outputs
    --experiment synthetic_blcs_v2 --run-id i932-select-s0-600-20260927 --scenes 600 --scene-offset 0 --seed 0 (holdout adds
    --scenes 400 --scene-offset 600 --seed 1 --fixed-thresholds <selection>/report.json)
parents: []
relations: []
papers: []
tags:
- synthetic_benchmark
- threshold_selection
---

## 要約

ballだけを証拠にするside判定（`src/tasks/court_side`、#932）の閾値を、物理cameraが既知のBLCS合成rallyで決めた。
選定（600 scene）と、選定に使わないscene・乱数seedでのheld-out（400 scene）の両方で、28条件すべての誤判定は0だった。
採用した閾値は `min_frames=8, max_cost=0.8, min_support=0.2, min_margin=0.15`（`min_motion_px=5`、`reprojection_px=20` @1080p）。
以前の閾値（Meiji clip_000の1例で決めた 0.5 / 0.5 / 0.1、重複除去なし）と比べ、誤検出・同期ずれ・校正誤差での停止率が大きく下がった。

## 条件と結果

条件は名目（欠落10%、pixel雑音2 px、校正雑音は正解仮説のcost中央値がMeiji clip_000の実測0.10に近い大きさ、window 300 frame、3 camera）から1因子ずつ変えた。
値は「誤判定率 / 停止率」。held-outの「以前」は同じ観測を以前の方式で判定した値。

| 条件 | 選定 (seed 0) | held-out (seed 1) | held-out 以前の方式 |
|---|---|---|---|


## メトリクスの解釈

- 観測: 誤判定は margin で止まっている。誤った仮説が最小costになった試行（選定 194件、held-out 136件）の margin は最大 0.103 / 0.088 で、採用した 0.15 との差は約0.05。
  いずれも最良の cost が 0.6 以上の、校正誤差 x4・同期 4 frame・誤検出30%以上といった強い劣化条件だった。
- 観測: cost と support の閾値（0.8 / 0.2）は緩い。誤判定を防いでいるのは margin で、cost/support は極端に不整合な clip を止める役割に限られる。
- 観測: 停止の主因は、欠落 70% 以上と window 30 frame（1秒）では `insufficient_frames` と margin 不足、誤検出・同期ずれ・校正誤差では margin 不足。
- 重複除去: 最初の試行（重複除去なし、seed 1 の scene 0-299）で、誤検出50%の条件に margin 0.25 の誤判定が1件あった。静止した誤検出の対（約80 frame）が誤った仮説に整合していた。
  直前に残したframeと観測viewが同じで、全viewの移動が 5 px 未満のframeを証拠から除く変更（b66db664）の後は、同種の誤判定は出ていない。

## 限界

- 合成の摂動は1因子ずつで、組み合わせは `combined`（欠落30%・誤検出10%・同期1 frame・雑音3 px）だけ。
- 静止した誤検出のjitterは観測雑音と同じ大きさにした。実検出器の誤検出の時間構造（trajectory gate の後に何が残るか）は測っていない。
- camera配置はBLCS生成器の分布（各端に2台、高さ3〜4 m）で、Meijiの配置そのものではない。
- 実clipでの検証は別 run（Meiji 全clip、検出器ballと注釈ball）で行う。

## 次に有効な実験

- Meiji 全clipで、検出器ballと注釈ballの判定を比べる（queue job `i932-court-side-meiji-clips-20260927`）。
- 実検出器の停止が多い場合は、停止理由ごとに、検出の欠落・静止した誤検出・同期ずれのどれが効いているかを分解する。
