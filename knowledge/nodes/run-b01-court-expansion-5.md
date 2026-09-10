---
id: run-b01-court-expansion-5
type: run
title: B01：SfM外周5%拡張の境界・描画検証
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: done
config:
  sfm_boundary_expansion_percent: 5.0
  sfm_boundary_margin_m: 0.5
metrics:
  expansion_percent: 5.0
  proposals: 2368
  outside_original_hull: 25
  maximum_original_hull_overshoot_m: 0.33871872867324004
  minimum_expanded_hull_clearance_m: 0.7461791845748778
  selected_for_render: 132
  rendered_frames: 118
  exterior_rendered_frames: 23
  exterior_ambiguous_frames: 2
  accepted_rendered_frames: 118
repro:
  commit: 96b98a17ee067d9956cafcfb65448d527910349b
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python outputs/b01-court-expansion-5/render_check.py
artifacts:
  run_dir: knowledge/runs/run-b01-court-expansion-5
  output_dir: /home/kamimura/projects/tennis-lab-worktrees/court-sfm-bounded/outputs/b01-court-expansion-5
parents:
- run-b01-court-shapes-balanced
relations: []
tags:
- court
- synthetic-data
- sfm-expansion
---

## 考察 / Findings

SfM水平凸包を頂点平均から1.05倍に拡大し、その後0.5m内側へ余白を取った。q90半径上限も同率拡大。複数コートの軌道中心に依存しない拡張で、0%は従来挙動を保持する。

2,368候補中25視点が元の外周外で、最大超過0.339m。拡張後の外周には全候補が収まり、最小余白0.746m。許容上限であり、5%の位置まで必ず使うわけではない。

外側25視点と各27グループ4代表を重複除去し132候補を選択。14候補が描画前の意味的条件で除外され、118枚をpublic NHT rendererで描画した。外側の除外はnear/far不定の2枚で、残る23枚をすべて描画。描画後追加除外0、全27グループ・14KPクラス・full/near_full/partialを確認した。全2,368候補を描画した実験ではない。

代表画像と外側視点を確認した。コート線が確認できる一方、ネットや遠景のぼけ・浮遊物は残る。可視性判定の通過は画像品質の完全保証ではない。下流精度比較は未実施。

前回の形状拡張実験に対し、今回の変更は境界拡張の明示設定。負数・非有限値・境界設定なしの指定を拒否し、全形状の0/5/15%包含と半径拡大をテストした。

B01のhuman_confirmed geometryを前回同様に明示的adapterで解釈した。canonical alignmentやデータセット公開は行っていない。共有training queueを使用した。
