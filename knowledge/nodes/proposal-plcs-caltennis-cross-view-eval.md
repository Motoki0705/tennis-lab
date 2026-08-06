---
id: proposal-plcs-caltennis-cross-view-eval
type: proposal
title: PLCSをCalTennisのcross-view consistencyで外部評価する
curator: chatgpt-schedule
date: 2026-08-06
status: ready
task: plcs
repo_paths:
  - src/tasks/plcs/data/dataset.py
  - src/tasks/plcs/training/metrics.py
  - src/tennis_scene/pipeline/components/plcs.py
evidence_runs: []
hypothesis:
  statement: 既存PLCS checkpointは単眼GVHMR translationを直接court座標へ変換するbaselineより、camera pair間のpositionとyawの不一致を低減する
  expected_effect: 固定20 clipsでmedian pairwise translation disagreementを20%以上低下させ、pelvis-relative pose disagreement悪化を5%以内、invalid率を10%以下にする
  failure_condition: translation disagreement改善10%未満、invalid率10%超、camera座標変換不能、またはmetric定義がpaper protocolと整合しない
evaluation:
  metrics: [pairwise_translation_disagreement_m, yaw_disagreement_deg, pelvis_relative_pose_disagreement_m, invalid_clip_rate]
  baseline_nodes: [run-i518-baseline]
  seeds: 1
  acceptance: 固定20 clipsでpairwise_translation_disagreement_mをbaseline比20%以上低下させ、pelvis_relative_pose_disagreement_mの悪化を5%以内、invalid_clip_rateを10%以下にする
parents:
  - run-i518-baseline
relations:
  - to: paper-arxiv-2606-20542
    rel: derived-from
tags: [literature, plcs, caltennis, external-evaluation]
---

## 背景

PLCSの既存runは主に合成datasetの教師3Dに対するposition・rotation errorで評価されている。CalTennisは実tennisを複数viewで収録し、absolute MoCapを用いずにcross-view consistencyを測るため、real-domain外部評価として補完的である。

## 現行実装との差分

CalTennis miniのtimestamps、camera calibration、player trackを`SceneDataset`へ適応し、camera pairごとに既存PLCSを実行する。`PLCSMetrics`へtranslation、yaw、pelvis-relative poseのview間不一致とinvalid clip率を追加する。model再学習は行わない。

## 最小検証

3 views以上を持つ同期区間から20 clipsを事前固定する。baselineは各cameraのGVHMR global translationを提供extrinsicsでcourt座標へ変換した値、treatmentは同じ2D poseとcourt evidenceを既存PLCSへ入力した値とする。

## 比較対象

既存graph上の比較起点として`run-i518-baseline`を参照する。runのsynthetic metricとCalTennisのcross-view metricは直接同値ではないため、既存値を再解釈せず、同checkpointまたは再現可能な現行baselineをCalTennis上で両条件に通す。

## 合格条件と停止条件

frontmatterのacceptanceを満たせば、対象clipsとcamera数を拡張する。20 clips終了時に改善10%未満、invalid率10%超、SMPL/COCO変換やcamera座標変換が成立しない場合は停止する。

## リスク

cross-view agreementはabsolute accuracyではなく、全view共通biasを検出できない。datasetの非商用条件、容量、privacy、SMPL-Xとtennis-lab contract差がある。metric実装はpaper定義を再現し、都合のよいsubset選択を避ける。
