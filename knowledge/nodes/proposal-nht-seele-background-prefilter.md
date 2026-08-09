---
{
  "id": "proposal-nht-seele-background-prefilter",
  "type": "proposal",
  "title": "NHT static backgroundへview-dependent Gaussian prefilterを導入する",
  "curator": "chatgpt-schedule",
  "date": "2026-08-07",
  "status": "candidate",
  "issue": 711,
  "task": "synthetic_data_generation",
  "repo_paths": [
    "src/synthetic_data_generation/dataset/plcs/rendering/nht.py",
    "src/synthetic_data_generation/dataset/blcs/rendering/nht.py",
    "src/synthetic_data_generation/dataset/court/rendering/nht.py",
    "src/synthetic_data_generation/rendering/nht/composition_smoke.py"
  ],
  "hypothesis": {
    "statement": "静的court/background Gaussianだけをcamera-cluster依存のactive-index集合へ絞ると、NHT appearance/AOV契約を維持したままdataset rendering latencyとVRAMを削減できる",
    "expected_effect": "固定100 frameでp50 frame timeを1.5倍以上高速化し、peak VRAMを20%以上削減する",
    "failure_condition": "LPIPS増加0.01超、SSIM低下0.005超、instance mask IoU低下0.5 point超、alpha consistency error 0.005超、連続background hole、またはp95 latency 10%以上悪化"
  },
  "evaluation": {
    "metrics": [
      "p50_frame_time_ms",
      "p95_frame_time_ms",
      "peak_vram_gb",
      "active_gaussians",
      "lpips_delta",
      "ssim_delta",
      "depth_mae",
      "instance_mask_iou",
      "aov_alpha_consistency_max_error"
    ],
    "baseline_nodes": [],
    "seeds": 3,
    "acceptance": "同一asset・camera・人物配置で3 seedsを実行し、p50を1.5倍以上高速化、peak VRAMを20%以上削減しつつLPIPS増加<=0.01、SSIM低下<=0.005、mask IoU低下<=0.5 point、alpha consistency error<=0.005を満たす"
  },
  "evidence_runs": [],
  "parents": [],
  "relations": [
    {
      "to": "paper-arxiv-2503-05168",
      "rel": "derived-from"
    }
  ],
  "tags": [
    "literature",
    "synthetic-data-generation",
    "nht",
    "rendering-optimization"
  ]
}
---

## 背景

SeeLeはview clusterごとに高寄与Gaussianを選ぶことで3DGSの処理量を削減する。tennis-labでは標準SH rasterizerではなくNHT feature/depth/instance AOVを扱うため、kernel置換ではなく入力Gaussian subsetの制御から検証する。

## 現行実装との差分

静的backgroundだけを24 camera clusterへ分け、calibration viewからshared/exclusive active indexを生成する。各frameでは最近傍clusterと近傍clusterの集合だけを既存rasterizationへ渡す。人物Gaussian、NHT feature、appearance_space_sha256、instance_id、shader、AOV pathは変更しない。

## 最小検証

同一background asset、2人物asset、default/broadcast由来の固定100 frame camera sequenceをbaseline/treatmentで描画する。100% Gaussian baselineとcluster prefilterを比較し、画質、depth、mask、AOV整合、latency、VRAMを測る。

## 比較対象

正式baseline runが未登録のため`status: candidate`とする。baseline runをformal graphへ登録してから`ready`へ進める。

## 合格条件と停止条件

frontmatterのacceptanceを満たす場合だけopacity-aware tile filtering等の次段へ進む。visibility hole、AOV不整合、人物mask変化、p95悪化が出た場合は停止し、動的avatarへclusterを適用しない。

## リスク

NHT rendererとSeeLeの標準3DGS rasterizerには契約差がある。上流kernelのlicense境界もあるため、初回は独立実装のindex prefilterに限定する。camera distributionに偏ったclusterは未見viewの背景欠損を生む可能性がある。
