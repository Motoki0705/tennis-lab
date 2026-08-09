---
{
  "id": "paper-arxiv-2603-16447",
  "type": "paper",
  "title": "ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars",
  "curator": "chatgpt-schedule",
  "date": "2026-08-07",
  "status": "reviewed",
  "external_ids": {
    "doi": null,
    "arxiv": "2603.16447",
    "openreview": null
  },
  "published_at": "2026-03-18",
  "reviewed_at": "2026-08-07",
  "evidence_level": "fulltext",
  "tasks": [
    "synthetic_data_generation",
    "cross_cutting"
  ],
  "repo_paths": [
    "src/synthetic_data_generation/dataset/plcs/components/avatar_control.py",
    "src/synthetic_data_generation/dataset/plcs/rendering/nht.py",
    "src/synthetic_data_generation/composition/gaussians.py",
    "src/synthetic_data_generation/composition/contracts.py"
  ],
  "sources": [
    {
      "kind": "publisher",
      "url": "https://openaccess.thecvf.com/content/CVPR2026/html/Song_ProgressiveAvatars_Progressive_Animatable_3D_Gaussian_Avatars_CVPR_2026_paper.html"
    },
    {
      "kind": "paper",
      "url": "https://arxiv.org/abs/2603.16447"
    },
    {
      "kind": "project",
      "url": "https://ustc3dv.github.io/ProgressiveAvatars/"
    }
  ],
  "relations": [],
  "tags": [
    "literature",
    "gaussian-avatar",
    "lod",
    "systems-optimization"
  ]
}
---

## 要約

ProgressiveAvatarsは、tracked FLAME meshのface-local Gaussianを画面空間gradientに応じて階層的に細分化し、最粗粒度のcoverageから重要度順にGaussianを追加する。独立LOD assetを切り替えず、転送量または描画予算に応じて品質を連続的に改善する。

## 主要な主張と根拠

著者はNeRSemble上で、full assetの一部だけでも描画可能なbase coverageを保ち、予算増加に応じてPSNR・SSIM・LPIPSを改善すると報告する。CVPR 2026公式本文、arXiv全文、公式projectを確認した。レビュー時点で公式codeは公開予定の状態である。

## tennis-labへの適用可能性

現行PLCS avatar controlはSMPL triangleへのbarycentric attachmentを持つが、hierarchy、importance、budget、base coverageをasset contractへ保存しない。既存attachmentとNHT appearance hashを維持したまま、active Gaussian subsetだけを25%、50%、100%へ変える比較が可能である。

## 制約・失敗条件

原手法は校正済み16 viewの頭部avatarとFLAMEを対象とする。全身SMPL、手足、racket、速いtennis motion、複数人物への一般化は未証明である。broadcast camera由来のimportanceは特定viewへ偏り、低予算で手足やracket周辺にcoverage holeを生む可能性がある。

## コード・データ・ライセンス

論文本文は公開されているが、レビュー時点で公式codeとそのlicenseは未確定である。NeRSemble、FLAME、SMPLの条件も独立に適用される。production導入は公式license確定と独立実験の通過後に限る。
