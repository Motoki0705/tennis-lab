---
id: paper-arxiv-2606-20542
type: paper
title: "CalTennis: Large Multi-View Tennis Video Dataset and Benchmark of Monocular-to-3D Pose Estimation"
curator: chatgpt-schedule
date: 2026-08-06
status: reviewed
external_ids:
  doi: null
  arxiv: "2606.20542"
  openreview: null
published_at: 2026-06-24
reviewed_at: 2026-08-06
evidence_level: fulltext
tasks: [plcs, scene_alignment, tennis_scene]
repo_paths:
  - src/tasks/plcs/data/dataset.py
  - src/tasks/plcs/training/metrics.py
  - src/tennis_scene/pipeline/components/plcs.py
  - src/synthetic_data_generation/scripts/alignment/calibrate_court_alignment.py
sources:
  - kind: paper
    url: https://arxiv.org/abs/2606.20542
  - kind: project
    url: https://ilonadem.github.io/caltennis-website/
  - kind: dataset
    url: https://huggingface.co/datasets/demalenk/caltennis
relations: []
tags: [literature, plcs, tennis, multi-view, benchmark]
---

## 要約

CalTennisは40 players、51 hours、約1,103万framesのtennis映像を2〜6台の同期cameraで60 Hz収録した実環境benchmarkである。court line交点からcamera extrinsicsを推定し、時刻offsetを最適化して単眼3D人体推定を共通court座標へ変換する。

## 主要な主張と根拠

5手法の比較から、absolute depth、ground contact、body shapeのview一貫性が主要な失敗点であると報告する。cross-view agreementはabsolute ground truthではなく、全view共通biasを検出できない。

## tennis-labへの適用可能性

PLCSは主に合成教師3Dで評価されるため、CalTennis miniを用いたcamera pair間position、yaw、pelvis-relative pose、invalid率はreal-domain外部評価になる。model再学習を伴わない最小検証が可能である。

## 制約・失敗条件

固定tripod・特定端末・court条件へのdomain依存がある。SMPL-X benchmarkとCOCO17/SMPL-H contractの変換が必要であり、cross-view一致だけでabsolute accuracyを主張してはならない。

## コード・データ・ライセンス

paper、project、Hugging Face datasetを確認した。datasetはCC BY-NC 4.0とされ、full splitは約122 GBである。公式evaluation codeが未公開の場合はpaper protocolを独立実装し、映像のprivacyと再配布条件を確認する。
