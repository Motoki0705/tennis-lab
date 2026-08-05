---
id: proposal-blcs-arc-physics-refinement
type: proposal
title: BLCS軌道へarc単位の最小物理refinementを導入する
curator: chatgpt-schedule
date: 2026-08-06
status: ready
task: blcs
repo_paths:
  - src/tasks/blcs/models/components/differentiable_projection.py
  - src/tasks/blcs/generate_dataset/simulation/ball_physics.py
  - src/tennis_scene/pipeline/components/blcs.py
evidence_runs: []
hypothesis:
  statement: 現行BLCS予測をcontact・bounce間arcへ分割し、再投影誤差でfitした最小物理modelを後処理すると、未補正軌道より位置または着地点誤差が低下する
  expected_effect: 3 seeds平均でposition errorまたは着地点誤差がbaseline比5%以上改善し、fitgがdrag+Magnusと同等なら単純なmodelを採用できる
  failure_condition: 改善2%未満、arc失敗率2%超、parameterの20%以上が許容境界へ張り付く、または処理時間が1秒/arcを超える
evaluation:
  metrics: [position_error, landing_point_error_m, reprojection_error_px, arc_failure_rate, wall_time_ms_per_arc]
  baseline_nodes: [run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep]
  seeds: 3
  acceptance: 3 seeds平均でposition_errorまたはlanding_point_error_mをbaseline比5%以上改善し、arc_failure_rateが2%以下、wall_timeが200 ms/arc以下であること
parents:
  - run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep
relations:
  - to: paper-title-2026-7aae71f6955e6061
    rel: derived-from
tags: [literature, blcs, physics, trajectory-refinement]
---

## 背景

現行BLCSは予測軌道にsmoothnessと固定gravityの学習lossを持つが、観測からdrag、Magnus、有効重力を同定していない。一方、微分可能projectionとball physics simulatorは既に存在する。学習contractを変更する前に、推論後のarc refinementで必要なmodel複雑度を測る。

## 現行実装との差分

`BLCSModule`出力をcontact・bounce境界で分割するadapterと、A:固定重力放物線、B:fitg、C:drag+Magnusの3 optimizerを追加する。全条件で同じ初期軌道、可視camera、projection loss、parameter boundを使う。学習済みcheckpointは固定する。

## 最小検証

固定test splitから最初の200有効arcを抽出し、2D noise・dropout強度別に3条件を評価する。contact/bounce ground truthが無い実映像はpilotに含めず、まず合成sceneの既知境界で同定可能性と計算量を確認する。

## 比較対象

baselineは`run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep`の未補正`ball_3d`である。3物理条件を同一予測から開始し、複雑modelの追加parameter以外を固定する。

## 合格条件と停止条件

合格条件はfrontmatterのacceptanceに従う。最初の200 arcで全条件の改善が2%未満、fit parameterの20%以上がboundへ張り付く、NaN/OOM、または1秒/arc超なら停止する。fitgがdrag+Magnusと同等以上ならMagnus fittingを学習lossへ導入しない。

## リスク

arc境界誤り、短区間の非同定、synthetic-to-real空力差、後処理latencyが主なriskである。論文のサッカー結果をテニスへ直接一般化せず、model rankingをtennis-lab条件で再測定する。
