---
id: paper-arxiv-2601-01660
type: paper
title: Animated 3DGS Avatars in Diverse Scenes with Consistent Lighting and Shadows
curator: chatgpt-schedule
date: 2026-08-07
status: reviewed
external_ids:
  doi: null
  arxiv: "2601.01660"
  openreview: null
published_at: 2026-01-04
reviewed_at: 2026-08-07
evidence_level: fulltext
tasks: [synthetic_data_generation, tennis_scene]
repo_paths:
  - src/synthetic_data_generation/composition/gaussians.py
  - src/synthetic_data_generation/rendering/nht/composition_smoke.py
  - src/synthetic_data_generation/dataset/plcs/rendering/nht.py
sources:
  - kind: paper
    url: https://arxiv.org/abs/2601.01660
  - kind: project
    url: https://miraymen.github.io/dgsm/
relations: []
tags: [literature, 3dgs, shadows, relighting]
---

## 要約

Deep Gaussian Shadow Mapsは異方性Gaussianの光線上積分からtransmittance atlasを構築し、animated Gaussian avatarを静的3DGS sceneへsoft shadow付きで合成する。scene cubemapから球面調和照明も推定する。

## 主要な主張と根拠

著者はAvatarX・ActorsHQのavatarと複数の3DGS sceneを用い、meshingなしでshadowとscene-matched relightingを生成できると報告する。公式全文とprojectを確認した。公式codeはレビュー時点でComing Soonである。

## tennis-labへの適用可能性

現行compositionはbackgroundとmovable Gaussianを結合するがshadow transmittanceを持たない。appearanceを変更せずshadow-only AOVを追加し、classical shadow mapをpseudo-GTに比較すれば、NHT relightingと分離した最小検証になる。

## 制約・失敗条件

屋外courtの太陽・空・時間変化は少数lightで近似しにくい。広いcourtではatlasのVRAM・更新時間が増える。NHT latent appearanceへSH scaleを直接適用する意味は保証されないため、shadowとrelightingを同時導入しない。

## コード・データ・ライセンス

公式codeとlicenseは未公開である。本文の数式から再実装する場合も、将来公開される実装との一致を検証し、上流3DGS sceneやavatar assetの利用条件を個別に管理する。
