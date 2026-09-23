---
id: run-tennis-scene-meiji-court-filter-5m-20260922
type: run
task: tennis_scene
sequence: 7
recorded_at: '2026-09-23'
title: 'モデルCourtによる人物選別: 5m baseline marginで遠方選手の観測が欠落'
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
  court_source: fresh model prediction from this task
  sideline_margin_m: 1.0
  baseline_margin_m: 5.0
  track_selection: auto
metrics:
  court_input: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/generate/responsibility_cleanup/20260922T191934Z-pipeline/court_kp_result.json
  court_sha256: e193110c5e332834631237a826104be4cf0615c0cb4a5472ccd31ba83f784923
  filter: existing court_footpoint_filter enabled; margins 1m sideline / 5m baseline;
    model-derived Court only
  records:
  - camera: cam0
    seconds: 312.8618838789989
    polygon_px:
    - - 1013.6500244140625
      - 358.87127685546875
    - - 1241.7001953125
      - 334.8659362792969
    - - 852.569091796875
      - 938.6590576171875
    - - 210.9071502685547
      - 616.5291137695312
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
    seconds: 305.47398695599986
    polygon_px:
    - - 289.8787841796875
      - 523.3621826171875
    - - 712.0733642578125
      - 492.8092041015625
    - - 2243.991943359375
      - 718.98779296875
    - - 785.984619140625
      - 1876.3182373046875
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
      - 2.858095645904541
    - - 10.375598907470703
      - 0.0
    matched_bbox_median:
    - 0.0
    - 0.0
    matched_bbox_p95:
    - 0.0
    - 0.013167819008231163
    opposite_order_better_frames: []
    observed_counts:
    - 1010
    - 940
  - camera: cam2
    seconds: 305.50307176099886
    polygon_px:
    - - 3175.67333984375
      - 1306.9114990234375
    - - -1963.234619140625
      - 1532.2940673828125
    - - 711.2093505859375
      - 395.3953552246094
    - - 1109.581787109375
      - 392.3114318847656
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
      - 0.8372650742530823
    - - 4.1237311363220215
      - 0.21369752287864685
    matched_bbox_median:
    - 0.0
    - 0.21369752287864685
    matched_bbox_p95:
    - 0.0
    - 0.9612497091293335
    opposite_order_better_frames: []
    observed_counts:
    - 1010
    - 298
repro:
  commit: fa8798a4c19f7efc6b123f4819130d57fe14b565
  branch: codex/tennis-scene-responsibility-cleanup
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup/.venv/bin/python
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T191934Z/tracker_court_filter_probe.py
artifacts:
  run_dir: knowledge/runs/run-tennis-scene-meiji-court-filter-5m-20260922
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790113108886965545_1794995_tennis-scene-cleanup-court-filter-tracking-20260922T2134Z.log
parents:
- run-tennis-scene-meiji-auto-person-mismatch-20260922
relations: []
papers: []
tags:
- identity_audit
- court_filter
- negative_result
---

## 観測

現行DINOを3camera×1010frameで実行し、推定Courtからの既存footpoint filterを有効化した。
cam0の両bbox軌跡は旧人手対応の対象と全要素一致した。cam1も中心差の中央値は両人0で、95%点は最大0.0132bbox幅だった。
cam2では対応候補そのものは同じ対象選手だったが、遠方選手の直接観測は298/1010frameだけとなった。
[拡大画像](../../runs/run-tennis-scene-meiji-court-filter-5m-20260922/evidence/tracking/cam2_far_crops.jpg)では、後半の推定bboxが実際の人物へ追従せず停止することを確認した。
数値上の旧bboxへの最近傍と、現在の映像を追跡できていることは区別する必要がある。

## 原因の切り分け

モデルHで旧人手対象のbbox足元をcourt平面へ戻す診断では、cam2遠方選手の|y|は最大約19.57mだった。
半長11.885m＋5mの領域では含まれないframeがあった。これは推定Hに基づく包含診断で、実測3D位置ではない。
さらにbaseline marginを8m以上へ広げると、一部cameraの前側cornerがcamera平面を越え、
4cornerをそのまま射影する既存実装のpolygonが反転した。ROIを広げるだけでは解消しない。

## 次の確認と限界

画像上の逆Hによる半平面をclipする実装へ修正し、可視領域を保持したままbaseline margin 10mで再確認する。
入力Courtはこのタスクで新規推論したもの。manual Courtと外注ballはfilter入力にしていない。
このrunは追跡のみで、scene・3D精度・最終の人手対応適用を完了扱いにしない。TensorBoardは学習なしのため対象外。
