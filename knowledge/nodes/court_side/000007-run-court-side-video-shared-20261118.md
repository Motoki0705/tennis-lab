---
id: run-court-side-video-shared-20261118
type: run
task: court_side
sequence: 7
recorded_at: '2026-11-18'
title: meiji_4cam全69 clipでの動画共有court sideと単独判定の照合
issue: 2241
provider: codex
date: '2026-11-18'
status: done
summary_impact: updated
config:
  dataset: meiji_4cam/video_000
  court_mode: video_shared
  court_side_mode: video_shared
  unobservable_policy: warn
  detector: X3D/CSN+DPT T32 BF16
  min_margin: 0.15
  max_cost: 0.8
  min_support: 0.2
  min_frames: 8
metrics:
  clips: 69
  scored_frames: 13869
  cost: 0.4307287246444468
  support: 0.6141754993150191
  margin: 0.2673329440506773
  standalone_agreement: 51
  conflicting: 0
  ambiguous_margin: 18
  shared_side_usable: 69
artifacts:
  run_dir: knowledge/runs/run-court-side-video-shared-20261118
  output_dir: .worktree-outputs/court-side-video-shared/tennis_scene/evaluate/court-side-20261118
parents:
- run-i932-synthetic-side-thresholds
- run-court-video-drift-lines-20261117
relations:
- rel: evaluates
  to: run-scene-x3d-dpt-20261116
papers: []
tags: []
repro:
  commit: ac77e4b7444d2ffe900a9ea858b85b16f38ca46c
  branch: codex/court-side-video-shared
---

## 固定4カメラの共有判定

meiji_4cam の video_000 では、cam0を基準にcam2とcam3を180度回す組合せが最良でした。69 clipすべてで単独の最良仮説もこの組合せです。単独で閾値を通ったのは51 clip、残る18 clipは次点とのmarginが0.15未満でした。全clipの証拠を統合すると、閾値を変えずに69/69 clipのcourt side nodeが通りました。[全69 clipの表](../../runs/run-court-side-video-shared-20261118/clips.csv)

| 単独判定と共有判定の関係 | clip数 | 共有モードでの扱い |
|---|---:|---|
| 確定して一致 | 51 | 採用 |
| 確定して矛盾 | 0 | 当該clipの失敗 |
| margin不足 | 18 | 理由を記録して共有判定を採用 |
| frame不足・カメラ対の非連結・絶対整合性不足 | 0 | 既定は理由付きwarning |

4カメラ各46,228元frameからclipごとに約30 fpsで計23,153 frameをサンプルし、静止観測を除いた13,869多視点frameで8仮説を採点しました。最良のcostは0.430729、supportは0.614175。次点はcam1も回す組合せでcost 0.698062、marginは0.267333です。6 camera対の共通frame数は5,907〜8,953でした。[動画の全8仮説とcamera対の件数](../../runs/run-court-side-video-shared-20261118/summary.json)

| clip | 有効frame | 単独margin | 単独判定 |
|---|---:|---:|---|
| 000 | 93 | 0.288507 | 一致 |
| 027 | 99 | 0.116264 | 未決 |
| 037 | 85 | 0.130303 | 未決 |

この3 clipの全単独仮説のcost・support・順位は #1752 の保存値と完全一致しました。別の計算で、clipごとに静止観測を除いた全frameを連結し、元のDLTで各仮説を直接採点しました。動画集計とのcost・supportの最大絶対差は1.11×10⁻¹⁶、使用frame数と全camera対の共通frame数は一致しました。[全552件の単独仮説](../../runs/run-court-side-video-shared-20261118/standalone-rankings.csv)

## 入力と実行範囲

#2191の共有Hと別保存のdrift診断をloadしました。#1752の3 clipではballと校正、過去のX3D検証のclip_018ではballを再利用しました。共有queueの排他予約で残る65 clip×4cameraのball検出・平滑化と計66 clipの共有コート校正を補完しました。GPUを使う前段は67分39秒で正常終了し、CUDA異常はありませんでした。CPUでのbuild、再読込、全clipの実node検証、独立再採点は16.74秒でした。CPU集計の初回はローカルの成果物root名が相対パスの先頭と衝突し、判定前に停止しました。集計スクリプトのrootを明示して解消しました。[入力の固定方法と再現手順](../../runs/run-court-side-video-shared-20261118/reproduce.md)

固定コートとX3D/CSN+DPTを使うこの動画について、#1752が共有判定を一括入口に接続してカメラ同期済みの全clipを処理するための、court side段の条件がそろいました。人物対応・三角測量・scene全体の69 clip完走は #1752 で検証する項目です。また本評価は幾何仮説の整合性を測っています。カメラ向きの正解ラベルに対する誤判定率や別の動画での性能は未測定です。
