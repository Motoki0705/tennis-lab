---
{
  "id": "proposal-ball-rstr-refinement-head",
  "type": "proposal",
  "title": "ball detectionへR-STR residual refinement headを追加する",
  "curator": "chatgpt-schedule",
  "date": "2026-08-07",
  "status": "candidate",
  "issue": 711,
  "task": "ball_detection",
  "repo_paths": [
    "src/tasks/ball_detection/models/input_adapter.py",
    "src/tasks/ball_detection/models/conv_next_unet.py",
    "src/tasks/ball_detection/training/lightning_module.py",
    "src/tasks/ball_detection/training/metrics.py"
  ],
  "hypothesis": {
    "statement": "MDD入力・ConvNeXt U-Net・loss・data splitを固定し、final logits後へR-STR型residual refinement headだけを追加すると、遮蔽および高速移動区間のfalse negativeが減少する",
    "expected_effect": "3 seeds平均でtest F1とRecallを各0.5 percentage point以上改善し、mean distance悪化を0.2 px以内、FPS低下を10%以内に保つ",
    "failure_condition": "OOMまたはNaN、FPS低下20%超、可変時系列長contract不整合、または3 seeds平均F1がbaseline以下"
  },
  "evaluation": {
    "metrics": [
      "test_f1",
      "test_recall",
      "occlusion_recall",
      "mean_distance_px",
      "false_negative_count",
      "fps",
      "peak_vram_gb"
    ],
    "baseline_nodes": [],
    "seeds": 3,
    "acceptance": "同一data・augmentation・optimizerで3 seeds平均test F1とRecallを各0.5 point以上改善し、mean_distance_px悪化0.2 px以内、FPS低下10%以内、16 GB GPU内にする"
  },
  "evidence_runs": [],
  "parents": [],
  "relations": [
    {
      "to": "paper-arxiv-2512-02789",
      "rel": "derived-from"
    }
  ],
  "tags": [
    "literature",
    "ball-detection",
    "spatiotemporal-refinement"
  ]
}
---

## 背景

現行ball detectionはMDD相当の入力を既に使うが、coarse heatmapを時空間contextで補正するresidual headを持たない。TrackNetV5全体を移植せず、未実装部分だけを分離する。

## 現行実装との差分

`ConvNeXtUNet`のfinal logits後にoptionalなR-STR型headを追加する。入力adapter、backbone、heatmap resize、Focal BCE、dataset、optimizerを固定する。公式proprietary SDKのcodeは使用しない。

## 最小検証

現行MDD設定、8 frame、既存splitとaugmentationをbaselineとする。patch size、layer数、head数、context dropoutを事前固定し、3 seedsで通常・occlusion・高速移動subsetを比較する。

## 合格条件と停止条件

frontmatterのacceptanceを満たした場合だけproduction設計へ進む。F1改善なし、計算量超過、可変T非対応、位置精度悪化で棄却する。

## リスク

原論文の3 frame・TrackNetV2・WBCE条件とは異なるため、報告利得を期待値として流用しない。全体attentionのtoken数とPixelShuffle復号が高解像度で支配的になる可能性がある。
