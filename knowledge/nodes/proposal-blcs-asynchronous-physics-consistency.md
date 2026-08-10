---
{
  "id": "proposal-blcs-asynchronous-physics-consistency",
  "type": "proposal",
  "title": "BLCSへdrag・Magnus・bounce整合性と非同期観測stress testを追加する",
  "curator": "chatgpt-schedule",
  "date": "2026-08-08",
  "status": "candidate",
  "issue": 720,
  "task": "blcs",
  "repo_paths": [
    "src/tasks/blcs/models/blcs_track_query_model.py",
    "src/tasks/blcs/training/tracking_losses.py",
    "src/tasks/blcs/training/tracking_metrics.py",
    "src/tasks/blcs/generate_dataset/simulation/ball_physics.py"
  ],
  "hypothesis": {
    "statement": "現行BLCS checkpoint/data契約を維持し、generatorと同型のdrag・Magnus・bounce residualを補助制約として追加すると、同期条件を悪化させずcamera time-shift/drop条件のtrajectoryとlanding予測を改善できる",
    "expected_effect": "非同期・長欠損stress subsetでlanding RMSEを10%以上改善し、同期position MAE悪化を2%以内に保つ",
    "failure_condition": "同期position MAEが5%以上悪化、非同期landing RMSE改善が5%未満、NaN/OOM、またはbounce前後でphysical residualが不安定"
  },
  "evaluation": {
    "metrics": ["position_mae_m", "trajectory_rmse_m", "first_bounce_landing_error_m", "second_bounce_landing_error_m", "long_gap_rmse_m", "physics_residual"],
    "baseline_nodes": [],
    "seeds": 3,
    "acceptance": "同一split・detector noise・checkpoint設定で3 seeds平均の同期position MAE悪化<=2%、非同期/欠損条件のlanding RMSE改善>=10%、長欠損trajectory RMSE改善>=10%を満たす"
  },
  "evidence_runs": [],
  "parents": [],
  "relations": [
    {"to": "paper-doi-10-1109-icra57147-2024-10610631", "rel": "derived-from"}
  ],
  "tags": ["literature", "blcs", "ball-physics", "asynchronous-observation"]
}
---

## 背景

BLCSのgeneratorにはdrag、Magnus、spin、bounceを含むBallPhysicsがあるが、tracking lossはsmoothnessとgravity中心である。ICRA 2024のfactor-graph論文は、非同期camera観測とこれらの物理状態を統合する直接的な先行例である。

## 現行実装との差分

GTSAM/iSAM2を最初から導入せず、既存BallPhysicsと同じ状態遷移からdifferentiable residualを作る。camera observationを人工的にtime-shift/dropするstress conditionを追加し、model/loss変更はphysics-consistency項に限定する。

## 最小検証

固定baseline checkpoint/config、同一synthetic val/test split、同一detector noiseを使う。同期、短shift、長gap、bounce跨ぎの固定subsetでposition/trajectory/landing errorを比較する。

## 比較対象

正式baseline runが未登録のため`status: candidate`とする。baseline runをformal graphへ登録した後に`ready`へ進める。

## 合格条件と停止条件

frontmatterのacceptanceを満たす場合だけvelocity/spin headやonline smootherへ進む。同期精度悪化、非同期改善不足、数値不安定が出た場合は停止する。

## リスク

原論文の実環境、camera timestamp、human-pose spin priorはtennis-labと異なる。初回はhuman-pose priorやGTSAM依存を導入せず、既存generator physicsとの表現整合だけを検証する。
