---
{
  "id": "proposal-court-jac-calibration-metric",
  "type": "proposal",
  "title": "court alignmentへJaC画像空間calibration metricを追加する",
  "curator": "chatgpt-schedule",
  "date": "2026-08-07",
  "status": "candidate",
  "issue": 711,
  "task": "scene_alignment",
  "repo_paths": [
    "src/synthetic_data_generation/alignment/components/evaluation/court_lines.py",
    "src/synthetic_data_generation/scripts/alignment/calibrate_court_alignment.py",
    "src/synthetic_data_generation/configs/alignment/calibrate_court_alignment.yaml"
  ],
  "hypothesis": {
    "statement": "semantic ITF line annotationを用いたJaC@2px/5pxをholdout metricへ追加すると、現行metric-space gateでは見落とす画像再投影のcalibration劣化を既知摂動に対して単調に検出できる",
    "expected_effect": "未摂動transformが全摂動familyで最高のmean JaC5となり、0.50m・1.0deg・2%の境界摂動でmean JaC5が各5 percentage points以上低下する",
    "failure_condition": "semantic element不足で順位が不安定、対称解が未摂動以上、または32 holdout viewの90%未満で境界摂動より未摂動JaC5が高い"
  },
  "evaluation": {
    "metrics": [
      "jac_2_px",
      "jac_5_px",
      "distance_weighted_q95_m",
      "template_coverage_fraction",
      "accepted_view_fraction"
    ],
    "baseline_nodes": [],
    "seeds": 3,
    "acceptance": "固定32 annotated holdout viewで未摂動transformが各centre/orientation/scale摂動familyの最高mean JaC5となり、0.50m・1.0deg・2%摂動で各5 point以上低下し、90%以上のviewで未摂動JaC5が境界摂動を上回る"
  },
  "evidence_runs": [],
  "parents": [],
  "relations": [
    {
      "to": "paper-doi-10-1109-cvprw63382-2024-00338",
      "rel": "derived-from"
    }
  ],
  "tags": [
    "literature",
    "scene-alignment",
    "camera-calibration",
    "metric"
  ]
}
---

## 背景

現行court alignmentはworld/court-spaceのline residual、coverage、transform stabilityを主に評価する。ProCCのJaCを画像空間holdout metricとして追加すれば、camera modelやscene transformの再投影誤差を別軸で診断できる。

## 現行実装との差分

fit solver、provider camera、scene_from_courtは変更しない。固定holdout viewへsemantic ITF line ID付きpolyline annotationを作り、court templateを画像へ投影してelement-level JaC@2px/5pxを計算するevaluation-only pathを`court_lines.py`へ追加する。

## 最小検証

未摂動selected transformに対しcentre shift 0.25/0.50m、orientation 0.5/1.0deg、relative scale 1/2%を個別に与え、JaCと既存q95/coverageを比較する。annotationはfitに使わない。

## 比較対象

正式baseline runが未登録のため`status: candidate`とする。既存alignment baselineをformal graphへ登録後に`ready`化する。

## 合格条件と停止条件

frontmatterのacceptanceを満たす場合だけholdout gateへの採用を検討する。対称解やpartial viewで既知摂動を一貫して識別できない場合はmetric導入を停止し、radial distortion等のcamera model変更へ進まない。

## リスク

semantic line annotation作業が必要で、少数可視lineではJaCが不安定になり得る。tennis courtの180度対称性はsoccerより強い曖昧性を生む。公式repository codeを直接転用せず、論文定義から独立実装する。
