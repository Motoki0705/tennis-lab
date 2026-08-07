---
id: paper-arxiv-2604-17567
type: paper
title: "Multi-Camera Self-Calibration in Sports Motion Capture: Leveraging Human and Stick Poses"
curator: chatgpt-schedule
date: 2026-08-06
status: reviewed
external_ids:
  doi: null
  arxiv: "2604.17567"
  openreview: null
published_at: 2026-04-24
reviewed_at: 2026-08-06
evidence_level: fulltext-code-data
tasks: [scene_alignment, plcs, tennis_scene]
repo_paths:
  - src/tennis_scene/pipeline/dependency_graph.py
  - src/tennis_scene/pipeline/orchestrator.py
  - src/tennis_scene/pipeline/components/gvhmr.py
  - src/tennis_scene/configs/pipeline.yaml
  - src/synthetic_data_generation/scripts/alignment/calibrate_court_alignment.py
sources:
  - kind: paper
    url: https://arxiv.org/abs/2604.17567
  - kind: project
    url: https://fandulu.github.io/sport_stick_multi_cam_calib/
  - kind: code
    url: https://github.com/fandulu/sport_stick_multi_cam_calib
  - kind: dataset
    url: https://github.com/fandulu/sport_stick_multi_cam_calib/tree/main/data
relations: []
tags: [literature, scene-alignment, camera-calibration, sports]
---

## 要約

同期RGB camera、既知intrinsics、人体2D keypoint、既知長用具の両端点からmetric-scale camera extrinsicsを自己校正する。essential matrixとmaximum spanning treeで初期poseを構築し、unscaled bundle adjustment、scale recovery、既知長とtemporal smoothnessを加えたscale-aware bundle adjustmentを行う。

## 主要な主張と根拠

3〜10 cameras、4 sports、5 noise levelsからなる160 synthetic sequencesで平均rotation error 0.020度、translation error 0.001 mを報告し、length constraint除去時に悪化する。主要定量結果はsyntheticで、実屋外golf例には定量ground truthがない。

## tennis-labへの適用可能性

GVHMRのhuman keypointにracquet端点観測を加え、court-only calibrationが弱いline dropout条件でcamera errorとdownstream PLCS errorを比較できる。camera metadataをPLCS前stage contractへ追加する必要がある。

## 制約・失敗条件

camera同期、既知intrinsics、複数viewで重なる人体・用具、正確な用具長が前提である。tennis racquet端点は遠景、motion blur、手による遮蔽で不安定で、個体長の誤差はscale biasになる。

## コード・データ・ライセンス

公式repositoryはMIT licenseでsynthetic dataと評価資産を公開する。三段optimizerのend-to-end再現経路は限定的で、数式からの独立実装が必要となる可能性がある。
