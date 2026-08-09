---
{
  "id": "paper-arxiv-2512-02789",
  "type": "paper",
  "title": "TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking",
  "curator": "chatgpt-schedule",
  "date": "2026-08-07",
  "status": "reviewed",
  "external_ids": {
    "doi": null,
    "arxiv": "2512.02789",
    "openreview": null
  },
  "reviewed_at": "2026-08-07",
  "evidence_level": "fulltext-code",
  "tasks": [
    "ball_detection"
  ],
  "repo_paths": [
    "src/tasks/ball_detection/models/input_adapter.py",
    "src/tasks/ball_detection/models/conv_next_unet.py",
    "src/tasks/ball_detection/training/lightning_module.py",
    "src/tasks/ball_detection/training/metrics.py"
  ],
  "sources": [
    {
      "kind": "paper",
      "url": "https://arxiv.org/abs/2512.02789"
    },
    {
      "kind": "code",
      "url": "https://github.com/codelancera-offical/TrackNetV5-SDK"
    }
  ],
  "relations": [],
  "tags": [
    "literature",
    "ball-detection",
    "spatiotemporal-refinement",
    "motion"
  ]
}
---

## 要約

TrackNetV5は、隣接frame差分を明化・暗化方向へ分離するMotion Direction Decoupling（MDD）と、粗いheatmapへ補正残差を加えるResidual-Driven Spatio-Temporal Refinement（R-STR）を提案する。R-STRはcontext dropout、patch embedding、時空間Transformer、PixelShuffle復号から構成される。

## 主要な主張と根拠

著者はTrackNetV2公開data上で高いF1・Recallを報告し、R-STRによってfalse negativeを大幅に減らしたと主張する。公式arXiv全文と公式SDKを確認した。ただし公開SDKのweightと社内学習dataは非公開で、報告条件を完全再現できるわけではない。

## tennis-labへの適用可能性

現行`input_adapter.py`はMDD相当のbrighten/darken入力とpower normalizationを既に持つ。一方、`conv_next_unet.py`は最終logitsを直接返し、R-STR相当のrefinement headを持たない。したがってMDDを再実装せず、final logits後のheadだけを追加する分離実験が可能である。

## 制約・失敗条件

原論文は3 frame、TrackNetV2 backbone、WBCE、二値ROI教師を用いる。tennis-labは8 frame MDD、ConvNeXt U-Net、Gaussian heatmap、Focal BCEであり、利得を直接移植できない。可変時系列長、VRAM、FPS、位置誤差を同時に検査し、平均F1が改善しない場合は採用しない。

## コード・データ・ライセンス

公式SDKはproprietary softwareであり、technical exchangeとacademic study向けとして公開されている。weightと学習datasetは非公開である。tennis-labへcodeをコピーせず、論文記述に基づく独立実装だけを検証対象とする。
