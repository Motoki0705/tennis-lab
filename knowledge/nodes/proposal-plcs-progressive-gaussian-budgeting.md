---
{
  "id": "proposal-plcs-progressive-gaussian-budgeting",
  "type": "proposal",
  "title": "PLCS avatarをprogressive Gaussian budgetで描画する",
  "curator": "chatgpt-schedule",
  "date": "2026-08-07",
  "status": "candidate",
  "issue": 711,
  "task": "synthetic_data_generation",
  "repo_paths": [
    "src/synthetic_data_generation/dataset/plcs/components/avatar_control.py",
    "src/synthetic_data_generation/dataset/plcs/rendering/nht.py",
    "src/synthetic_data_generation/composition/gaussians.py",
    "src/synthetic_data_generation/composition/contracts.py"
  ],
  "hypothesis": {
    "statement": "既存SMPL barycentric attachmentとNHT appearance spaceを固定し、全surfaceを覆うbase setと重要度順detail setへGaussianを階層化すると、random subsetより低予算画質と描画時間を改善できる",
    "expected_effect": "25% budgetで同数random subset比LPIPSを15%以上改善し、mask IoU低下を2 point以内、frame timeをfull budget比25%以上短縮する",
    "failure_condition": "連続したbody coverage hole、instance_id変化、AOV alpha consistency error 0.005超、またはhierarchy metadataがasset sizeを20%以上増加"
  },
  "evaluation": {
    "metrics": [
      "foreground_psnr",
      "foreground_ssim",
      "foreground_lpips",
      "person_mask_iou",
      "active_gaussian_count",
      "asset_bytes",
      "frame_time_ms",
      "aov_alpha_consistency_error"
    ],
    "baseline_nodes": [],
    "seeds": 3,
    "acceptance": "25% budgetでrandom比LPIPSを15%以上改善し、mask IoU低下2 point以内、full比frame time 25%以上短縮、全budgetでAOV alpha consistency error 0.005以下にする"
  },
  "evidence_runs": [],
  "parents": [],
  "relations": [
    {
      "to": "paper-arxiv-2603-16447",
      "rel": "derived-from"
    }
  ],
  "tags": [
    "literature",
    "plcs",
    "gaussian-avatar",
    "systems-optimization"
  ]
}
---

## 背景

PLCS rendererは人物assetの全Gaussianを常時composeする。SMPL triangle attachmentは既にあるが、base coverage、hierarchy、importance、budgetを表すcontractがない。

## 現行実装との差分

各Gaussianへparent face、hierarchy level、importance、stable index rangeを保存する。最粗粒度は全body surfaceを覆い、detail Gaussianは累積budgetに従ってactivateする。appearance hash、instance ID、depth・alpha AOV契約は変更しない。

## 最小検証

同一asset、同一50 frame、同一cameraで25%、50%、100% budgetを比較する。同数random順とface-uniform順をcontrolにし、full budget imageを参照する。

## 合格条件と停止条件

frontmatterのacceptanceを満たした場合にだけ階層学習へ進む。coverage hole、手足やracket周辺の欠落、label不整合、metadata肥大化が発生した場合は棄却する。

## リスク

原論文はFLAME頭部と16 cameraを対象とする。全身SMPLとbroadcast cameraではimportance分布が偏る可能性があり、pose変化に対して固定importanceが安定する保証もない。公式codeとlicenseが未公開なので、本文記述から独立に検証する。
