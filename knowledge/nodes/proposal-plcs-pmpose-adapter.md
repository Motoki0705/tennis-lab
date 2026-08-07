---
id: proposal-plcs-pmpose-adapter
type: proposal
title: PLCS前処理のViTPose-HをPMPose adapterと比較する
curator: chatgpt-schedule
date: 2026-08-07
status: ready
task: plcs
repo_paths:
  - src/submodules/models/vitpose/pose2d.py
  - src/tennis_scene/pipeline/components/gvhmr.py
  - src/tennis_scene/configs/pipeline.yaml
hypothesis:
  statement: detectorとbboxを固定してPMPose-Bへ差し替えると、occlusion条件の2D poseとvisibility calibrationが改善し、PLCS出力を悪化させない
  expected_effect: 全体OKS APを2 point、occlusion subsetを3 point以上改善し、PLCS position error悪化を2%以内に保つ
  failure_condition: 100 frame時点でocclusion subset改善0以下、far subset 2 AP以上悪化、dependency隔離不能、OOM、または処理時間2倍超
evaluation:
  metrics: [oks_ap, wrist_ankle_error_normalized, visibility_brier_score, gvh_mr_reprojection_error_px, plcs_position_error_m, velocity_outlier_rate, fps, peak_vram_gb]
  baseline_nodes: [run-i518-baseline]
  seeds: 3
  acceptance: 3 seeds平均で全体OKS APを2 point、occlusion subsetを3 point以上改善し、far subset低下1 point以内、PLCS position error悪化2%以内、処理時間2倍以内、16 GB GPU内にする
evidence_runs: []
parents:
  - run-i518-baseline
relations:
  - to: paper-arxiv-2601-15200
    rel: derived-from
tags: [literature, plcs, human-pose, occlusion]
---

## 背景

現行GVHMR前処理はtrack bboxをViTPose-Hへ渡し、COCO-17座標と単一confidenceを後段へ供給する。PMPoseは同じbbox条件でpresence、visibility、expected OKSを出力する。

## 現行実装との差分

`Pose2DRequest`と`Pose2DResult`を維持するPMPose adapterを隔離して追加する。detector、tracker、HMR2、GVHMR、PLCS checkpointは固定する。SAMを使う反復mask refinementは第二段階へ分離する。

## 最小検証

near、far、motion blur、partial occlusion、人物重なりを層別化した手動COCO-17付き300 frameを固定する。同一bboxをViTPose-HとPMPose-Bへ入力する。

## 比較対象

formal graphの`run-i518-baseline`をPLCS比較起点とする。2D pose adapter以外のpipeline構成とcheckpointを固定する。

## 合格条件と停止条件

frontmatterのacceptanceを満たせばmask条件付け段階へ進む。100 frameでocclusion gainなし、far subset大幅悪化、GPL依存を隔離不能、OOMで停止する。

## リスク

改善対象はcrowded scene中心で、通常tennisへのgainは限定され得る。GPL-3.0 codeを直接vendorせず、隔離runnerまたは独立adapterで評価する。presence/visibility出力の実checkpoint contractをテストする。
