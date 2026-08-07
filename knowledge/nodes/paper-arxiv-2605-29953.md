---
id: paper-arxiv-2605-29953
type: paper
title: "Mesh-Aware Epipolar Matching for Multi-View Multi-Person 3D Pose Estimation in Basketball"
curator: chatgpt-schedule
date: 2026-08-06
status: reviewed
external_ids:
  doi: null
  arxiv: "2605.29953"
  openreview: null
published_at: null
reviewed_at: 2026-08-06
evidence_level: fulltext-code
tasks: [plcs, tennis_scene]
repo_paths:
  - src/tennis_scene/pipeline/components/player_association.py
  - src/tennis_scene/pipeline/components/gvhmr.py
  - src/tennis_scene/pipeline/components/plcs.py
  - src/tasks/plcs/data/tracking_dataset.py
sources:
  - kind: paper
    url: https://arxiv.org/abs/2605.29953
  - kind: code
    url: https://github.com/Yinlipp/MAEM
relations: []
tags: [literature, plcs, multi-view, association, epipolar]
---

## 要約

MAEMは校正済み複数cameraのteam sports映像を対象とするtraining-freeのmulti-person 3D pose pipelineである。単眼mesh recoveryから人物bbox、2D/3D keypoint、画像へ投影したdense mesh vertexを取得し、bbox中心の再投影gate、対応vertexのepipolar距離gate、camera pairごとのHungarian matching、Union-Find clustering、keypoint単位のRANSAC triangulationを順に適用する。

## 主要な主張と根拠

論文はSportCenterでMPJPE/PA-MPJPE 59.8/40.7 mm、Human-M3 Basketballで74.0/51.8 mmを報告し、training-free association baselineよりMPJPEをそれぞれ9.0%と19.1%低下させたとする。改善はdense mesh projectionを用いた幾何filterを追加した対象benchmark上の結果であり、通常2選手で遠景が多いtennis映像への転移は未検証である。

## tennis-labへの適用可能性

現在の`PlayerAssociationModule`はcameraごとのGVHMR trackをmanual UIでcanonical player軸へ対応付ける。既存の出力契約を維持したまま、calibrated camera matrixとdense vertex projectionを入力する`automatic_geometry` modeを追加し、MAEM型の二段gateとmatchingを比較できる。manual assignmentとの一致率、ID switch、未割当率、PLCS downstream errorを測れば、手動工程を置換できるか反証可能である。

## 制約・失敗条件

既知intrinsic/extrinsicと、全cameraで対応するdense projected mesh topologyが必要である。論文記述のMHR 18,439 verticesと公開READMEの改変SAM 3D Bodyによる6,890 SMPL verticesにはfrontend contract差がある。処理はframe独立で時間的平滑化を持たず、severe occlusionやmotion blurでmesh自体が誤るとassociationも破綻する。公開実装のmatchingはCPU中心で、論文・READMEの条件ではframe当たり1秒超となり得る。

## コード・データ・ライセンス

公式repositoryはMIT licenseで、modified SAM 3D Body出力、SportCenter/Human-M3用matching、xrMoCap triangulation手順を公開する。ただしSAM 3D Body、xrMoCap、SMPL系body model、各datasetとcheckpointには個別の利用条件がある。コードを直接統合する場合も依存物のlicenseと配布条件を別途確認する。
