---
id: run-tennis-scene-meiji-visible-court-roi-20260922
type: run
task: tennis_scene
sequence: 8
recorded_at: '2026-09-23'
title: 可視ROIの投影修正で遠方選手の追跡を回復
provider: codex
session: 01a0c8c3-d190-7c51-b671-ded223340a2f
date: '2026-09-23'
status: done
config:
  frames: 1010
  camera_ids:
  - cam0
  - cam1
  - cam2
  court_source: fresh model Court from this task
  track_selection: auto
  sideline_margin_m: 1.0
  baseline_margin_m: 10.0
  roi_projection: inverse homography halfplanes clipped to visible image
metrics:
  court_input: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/generate/responsibility_cleanup/20260922T191934Z-pipeline/court_kp_result.json
  court_sha256: e193110c5e332834631237a826104be4cf0615c0cb4a5472ccd31ba83f784923
  filter: existing court_footpoint_filter enabled; visible image-clipped polygon,
    margins 1m sideline / 10m baseline; model-derived Court only
  records:
  - camera: cam0
    seconds: 311.64150579800116
    polygon_px:
    - - 0.0
      - 684.2243515419735
    - - 1040.4280194288738
      - 350.2762849866068
    - - 1246.431901864793
      - 327.5240906675274
    - - 761.4779459541057
      - 1080.0
    - - 0.0
      - 1080.0
    old_ids:
    - 1
    - 2
    new_ids:
    - 1
    - 2
    old_to_new_proposal:
    - 0
    - 1
    bbox_costs_median:
    - - 0.0
      - 4.359797477722168
    - - 12.638323783874512
      - 0.0
    matched_bbox_median:
    - 0.0
    - 0.0
    matched_bbox_p95:
    - 0.0
    - 0.0
    opposite_order_better_frames: []
    observed_counts:
    - 1010
    - 684
  - camera: cam1
    seconds: 304.99486693499784
    polygon_px:
    - - 285.51092185338393
      - 511.4503161923733
    - - 665.1941846589755
      - 485.887763655065
    - - 1920.0
      - 671.1523445384375
    - - 1920.0
      - 1080.0
    - - 493.9883799582731
      - 1080.0
    old_ids:
    - 1
    - 2
    new_ids:
    - 1
    - 2
    old_to_new_proposal:
    - 0
    - 1
    bbox_costs_median:
    - - 0.0
      - 2.860531806945801
    - - 10.375598907470703
      - 0.0
    matched_bbox_median:
    - 0.0
    - 0.0
    matched_bbox_p95:
    - 0.0
    - 0.0
    opposite_order_better_frames: []
    observed_counts:
    - 1010
    - 995
  - camera: cam2
    seconds: 305.1266216349977
    polygon_px:
    - - 0.0
      - 697.7284754049119
    - - 736.1935560166003
      - 384.77463344751175
    - - 1086.7189194211423
      - 382.19069911139195
    - - 1920.0
      - 751.0605632062325
    - - 1920.0
      - 1080.0
    - - 0.0
      - 1080.0
    old_ids:
    - 1
    - 2
    new_ids:
    - 1
    - 2
    old_to_new_proposal:
    - 0
    - 1
    bbox_costs_median:
    - - 0.0
      - 0.8701757192611694
    - - 4.1237311363220215
      - 0.0
    matched_bbox_median:
    - 0.0
    - 0.0
    matched_bbox_p95:
    - 0.0
    - 0.00013858973397873342
    opposite_order_better_frames: []
    observed_counts:
    - 1010
    - 925
repro:
  commit: fa8798a4c19f7efc6b123f4819130d57fe14b565
  branch: codex/tennis-scene-responsibility-cleanup
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup/.venv/bin/python
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T191934Z/tracker_court_filter_visible_probe.py
artifacts:
  run_dir: knowledge/runs/run-tennis-scene-meiji-visible-court-roi-20260922
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790114989277783677_1903278_tennis-scene-cleanup-visible-court-tracking-20260922T2207Z.log
parents:
- run-tennis-scene-meiji-court-filter-5m-20260922
relations: []
papers: []
tags:
- identity_audit
- court_filter
- visible_roi
---

## 変更と検証

拡張Courtの4cornerを直接射影する方法を、画像上の逆Hの半平面でclipする方法に変更した。
Court中心と同じ可視branchを選び、camera平面を越える拡張cornerがあってもROIが反転しない。
独立したpixel→world包含判定との比較を含む25件と、関連pipeline回帰87件が通過した。旧alignment実験専用のlocal_dataテスト1件は入力環境未指定でskipした。

モデルCourt、side margin 1m、baseline margin 10mで3camera×1010frameを新規追跡した。
全cameraで旧人手対応の対象2名へ一意な近接対応があり、逆対応が優勢になるframeは0だった。
cam0両名とcam1両名の中心差中央値・95%点は0。cam2遠方選手の95%点も0.000139bbox幅となった。

直接検出に裏付けられたframe数はcam0 [1010,684]、cam1 [1010,995]、cam2 [1010,925]。
5m marginとの比較でcam2遠方選手は298→925となり、[拡大画像](../../runs/run-tennis-scene-meiji-visible-court-roi-20260922/evidence/tracking/cam2_far_crops.jpg)でも後半の追従が回復した。
残る欠損は追跡補間を含み、全frameが直接観測であるとは扱わない。

## 解釈と次の実行

対象コートの領域制約が必要であり、track IDや累積面積順位だけでは既存の人手対応を再適用できなかった。
ROIを広げる際はprojective horizonをまたぐcornerの扱いも必要だった。
Courtはモデル出力だけを入力にし、旧bboxは人手対応の意味を照合する用途に限定した。manual Courtと外注ballは入力していない。

このrunはtrackingの確認で、scene全体の3D精度や姿勢推論の評価ではない。全段再生成時にはbboxと新規2D姿勢の全時系列を再照合してからassociationを適用する。
学習を行っていないためTensorBoardは対象外。
