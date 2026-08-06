---
id: paper-arxiv-2601-15200
type: paper
title: BBoxMaskPose v2: Expanding Mutual Conditioning to 3D
curator: chatgpt-schedule
date: 2026-08-07
status: reviewed
external_ids:
  doi: null
  arxiv: "2601.15200"
  openreview: null
published_at: 2026-01-21
reviewed_at: 2026-08-07
evidence_level: fulltext-code-data
tasks: [human_pose]
repo_paths:
  - src/submodules/models/vitpose/pose2d.py
  - src/tennis_scene/pipeline/components/gvhmr.py
  - src/tennis_scene/configs/pipeline.yaml
sources:
  - kind: paper
    url: https://arxiv.org/abs/2601.15200
  - kind: project
    url: https://mirapurkrabek.github.io/BBox-Mask-Pose/
  - kind: code
    url: https://github.com/MiraPurkrabek/BBoxMaskPose
  - kind: dataset
    url: https://huggingface.co/datasets/vrg-prague/OCHuman-Pose
relations: []
tags: [literature, human-pose, occlusion, visibility]
---

## 要約

BBoxMaskPose v2はperson detector、mask-conditioned PMPose、pose-guided segmentationを相互条件付けする。PMPoseはCOCO-17 keypointに加えてjoint presence、visibility、expected OKSを予測する。

## 主要な主張と根拠

著者は同一RTMDet-L bbox条件でPMPose-BがViTPose系baselineよりOCHuman-Pose APを改善し、COCO val APを概ね維持すると報告する。公式全文、project、code、weights、OCHuman-Pose datasetを確認した。

## tennis-labへの適用可能性

現行ViTPose adapterと同じbbox入力・Pose2D contractを維持してPMPoseだけを差し替えれば、detector、tracking、GVHMR、PLCS checkpointを固定した比較になる。presenceとvisibilityは画像外jointとoccluded jointを区別する入力契約へ接続できる。

## 制約・失敗条件

通常tennisはcrowded sceneが少なく、遠景playerはsmall instanceである。公式実装はGPL-3.0とMMPose/MMCV/MMDetection/SAM系依存を持つ。far subsetが2 AP以上悪化、occlusion subset改善なし、環境隔離不能、OOMで停止する。

## コード・データ・ライセンス

公式codeはGPL-3.0である。MIT repositoryへ直接vendorせず、隔離reference runnerまたは論文仕様に基づく独立adapterを用いる。OCHuman-Poseはannotation公開条件と元画像の利用条件を別々に確認する。
