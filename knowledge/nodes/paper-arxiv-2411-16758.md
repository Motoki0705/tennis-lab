---
id: paper-arxiv-2411-16758
type: paper
title: Bundle Adjusted Gaussian Avatars Deblurring
curator: chatgpt-schedule
date: 2026-08-07
status: reviewed
external_ids:
  doi: null
  arxiv: "2411.16758"
  openreview: null
published_at: 2024-11-25
reviewed_at: 2026-08-07
evidence_level: fulltext-code-data
tasks: [synthetic_data_generation, human_pose]
repo_paths:
  - src/synthetic_data_generation/dataset/plcs/components/avatar_control.py
  - src/synthetic_data_generation/dataset/plcs/rendering/nht.py
  - src/tennis_scene/pipeline/components/gvhmr.py
sources:
  - kind: paper
    url: https://arxiv.org/abs/2411.16758
  - kind: code
    url: https://github.com/MyNiuuu/MAD-Avatar
  - kind: dataset
    url: https://drive.google.com/file/d/1FXFILsI3WjxVL5ercZUHnSatL9dAbEib/view?usp=sharing
relations: []
tags: [literature, gaussian-avatar, motion-blur, human-pose]
---

## 要約

露光中の人体運動をB-spline軌道として表し、複数sub-frameへ変形したGaussian avatarのsharp renderを平均することで、blurred multi-view入力から鮮明なanimatable avatarと露光内poseを同時推定する。

## 主要な主張と根拠

著者は、単一時刻poseでblur画像を説明する代わりに3D blur formation modelとbundle adjustmentを用いることで、BlurZJUおよびBS-Human上のPSNR・SSIM・LPIPSとpose品質を改善すると主張する。公開本文、公式実装、配布datasetへの導線を確認した。報告値はcalibrated multi-view・単一人物条件に基づく。

## tennis-labへの適用可能性

現行PLCS synthetic rendererは1 frameにつき単一pose・単一rasterizationである。隣接SMPL transformをsub-frame補間し、RGBのみ平均しながら中央sub-frameのdepth・mask・3D poseを教師として保持する変更は、architecture変更と切り離した反証可能なrenderer ablationになる。

## 制約・失敗条件

複数player、racket、屋外背景、camera motion blur、rolling shutterは論文の検証外である。Gaussian assetが連続SMPL変形に対応しない場合、transform補間だけでは衣服の非剛体変形を再現できない。sub-frame数に比例して時間とVRAMが増える。

## コード・データ・ライセンス

公式codeとBlurZJU/BS-Humanへの配布導線を確認した。上流3DGS、各dataset、SMPLの利用条件は独立して確認する必要があり、生成assetの再配布可否をcode公開条件だけから推定してはならない。
