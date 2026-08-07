---
id: paper-arxiv-2605-21001
type: paper
title: "DAMA: Disentangled Body-Anchored Gaussians for Controllable Multi-Layered Avatars"
curator: chatgpt-schedule
date: 2026-08-07
status: reviewed
external_ids:
  doi: null
  arxiv: "2605.21001"
  openreview: J5VMAFJmho
published_at: 2026-05-26
reviewed_at: 2026-08-07
evidence_level: fulltext-code
tasks: [synthetic_data_generation, human_pose]
repo_paths:
  - src/synthetic_data_generation/composition/contracts.py
  - src/synthetic_data_generation/dataset/plcs/components/avatar_control.py
  - src/synthetic_data_generation/dataset/plcs/rendering/nht.py
sources:
  - kind: paper
    url: https://openaccess.thecvf.com/content/CVPR2026W/PhysHuman/html/Eskandar_DAMA_Disentangled_Body-Anchored_Gaussians_for_Controllable_Multi-Layered_Avatars_CVPRW_2026_paper.html
  - kind: project
    url: https://danieleskandar.github.io/dama/
  - kind: code
    url: https://github.com/danieleskandar/DAMA-code
relations: []
tags: [literature, gaussian-avatar, garments, human-pose]
---

## 要約

DAMAはskin・hair・upper・lower・outer等のGaussian layerを分離し、各meanをSMPL-X face上のbarycentric coordinateと正のnormal offsetへ分解して、横方向driftとbody penetrationを抑える。

## 主要な主張と根拠

著者は4D-DRESS 82 scansでChamfer distance 19.88 mm、penetration rate 1.46%、penetration depth 0.32 mmを報告する。一方、full-avatar PSNRは通常2DGSより低く、geometry制約と外観品質のtrade-offがある。CVPRW公式本文とMIT codeを確認した。

## tennis-labへの適用可能性

現行avatar controlはface indexとbarycentric embeddingを持つが、positive offset、semantic layer、layer orderを契約化していない。appearanceを固定してgeometry parameterizationだけを比較すれば、body penetrationとtemporal mask stabilityへの効果を分離できる。

## 制約・失敗条件

原手法はcalibrated multi-view、garment segmentation、SMPL-X fitを要求する。遠景、motion blur、racket occlusionでは入力誤差が増える。NHT deferred appearanceはRGB 2DGSと直接互換でなく、初期実験でappearance backendを変更してはならない。

## コード・データ・ライセンス

公式codeはMITである。4D-DRESS、SMPL-X、AMASSの利用条件は別途適用され、学習素材や生成assetの再配布可否を個別確認する必要がある。
