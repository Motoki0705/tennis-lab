---
id: proposal-plcs-subframe-motion-blur-avatar
type: proposal
title: PLCS synthetic rendererへ露光内sub-frame motion blurを導入する
curator: chatgpt-schedule
date: 2026-08-07
status: candidate
task: synthetic_data_generation
repo_paths:
  - src/synthetic_data_generation/dataset/plcs/components/avatar_control.py
  - src/synthetic_data_generation/dataset/plcs/rendering/nht.py
  - src/synthetic_data_generation/dataset/execution.py
hypothesis:
  statement: 隣接SMPL transformを露光内で補間し複数renderのRGBを平均すると、単一pose renderより実写の人物motion blur分布へ近づく
  expected_effect: 高速動作clipでLPIPSを10%以上改善し、中央sub-frame labelとのmask IoU低下を0.02以内に保つ
  failure_condition: LPIPS改善5%未満、人物centroid差5 px超、NaN、OOM、またはframe timeがbaselineの6倍を超える
evaluation:
  metrics: [lpips, ssim, optical_flow_distribution_distance, mask_iou, frame_time_ms, peak_vram_gb]
  baseline_nodes: []
  seeds: 3
  acceptance: 3 assets平均でLPIPSを10%以上改善し、mask IoU低下0.02以内、追加VRAM 2 GB以内、frame time 6倍以内にする
evidence_runs: []
parents: []
relations:
  - to: paper-arxiv-2411-16758
    rel: derived-from
tags: [literature, plcs, synthetic-data, motion-blur]
---

## 背景

現行PLCS NHT rendererは各frameで単一pose assetを一度だけrasterizeするため、露光中の連続運動を画像へ反映しない。formal graph上で同rendererの確定baseline runを特定できないため、状態は`candidate`とする。

## 現行実装との差分

前後frameのSMPL transformから5 sub-frameの軌道を生成し、各sub-frameを同一camera・appearanceでrenderする。RGBは平均し、depth、instance mask、3D poseは中央sub-frameを教師として保存する。camera blurとrolling shutterは同時に変更しない。

## 最小検証

serve、forehand、lateral motionを含む3 assetsと固定camera列を使い、露光時間1/120、1/60、1/30秒を比較する。実写高速動作20区間に対するLPIPS・SSIM、optical-flow分布とlabel整合性を測る。

## 比較対象

baselineは現行の1 pose・1 rasterization経路である。正式baseline runを登録後に`baseline_nodes`と`parents`を設定し、`ready`へ進める。

## 合格条件と停止条件

frontmatterのacceptanceを満たした場合のみdownstream pose学習へ進む。中央labelとのcentroid差5 px超、NaN、OOM、または改善5%未満なら停止する。

## リスク

連続変形可能でない離散avatar assetでは補間が外観破綻を生む。複数player、racket、camera motion blur、rolling shutterは別要因であり、同時導入すると因果帰属できない。
