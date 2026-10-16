---
id: run-i932-meiji-clips-side-20260927
type: run
task: court_side
sequence: 2
recorded_at: '2026-09-27'
title: 'court_side: Meiji 3cam全clipで検出器ballと注釈ballからside判定'
issue: 932
provider: claude
date: '2026-09-27'
status: done
config:
  dataset: data/tennis_multivew/processed/meiji_3cam/dataset (57 clips, 3 cameras, 1920x1080)
  observe: default pipeline.yaml, court_detection/calibration and ball_detection (ft-e13) only; person, body and GVHMR disabled
  observe_code: evaljob worktree 58b2b720 (court and ball config identical to campaign930/i932-2-component)
  observe_job: 1790463646826379176_1274968_i932-court-side-meiji-clips-20260927 (observe finished for all clips; the job's own
    decide stopped on the 1e-7 coordinate tolerance, fixed in #942)
  decide: CPU, campaign930/i932-2-component, --name decisions_v2
  court_side: reprojection_px 20, min_motion_px 5, min_frames 8, max_cost 0.8, min_support 0.2, min_margin 0.15
  reference: outsource/<camera>_annotations.json observed points, judged by the same court_side component
metrics:
  clips: 57
  observe_failed_court_detection: 6
  detector_decided: 7
  detector_stopped_ambiguous_margin: 41
  detector_stopped_disconnected_views: 2
  detector_stopped_no_consistent_hypothesis: 1
  annotation_decided: 50
  annotation_stopped_disconnected_views: 1
  both_decided: 6
  agreements: 6
  detector_wrong_decisions: 0
  detector_wrong_argmin_clips: 9
  detector_wrong_argmin_max_margin: 0.086
  detector_decided_margin_min: 0.153
  annotation_decided_margin_min: 0.342
  clip000_annotation_matches_human: true
  detector_ball_recall_cam0: 0.214
  detector_ball_recall_cam1: 0.487
  detector_ball_recall_cam2: 0.505
  detector_ball_precision_cam0: 0.459
  detector_ball_precision_cam1: 0.631
  detector_ball_precision_cam2: 0.757
artifacts:
  run_dir: knowledge/runs/run-i932-meiji-clips-side-20260927
  output_dir: outputs/court_side/evaluate/meiji_clips/i932-detector-v1-20260927
  command: 'observe (queue, GPU): PYTHONPATH=. .venv/bin/python tests/benchmarks/court_side_clips.py --repo <repo> --dataset <repo>/data/tennis_multivew/processed/meiji_3cam/dataset
    --report <repo>/outputs/court_side/evaluate/meiji_clips/i932-detector-v1-20260927 --phase observe --device cuda;
    decide (CPU): same with --phase decide --device cpu --name decisions_v2'
parents:
- run-i932-synthetic-side-thresholds
relations: []
papers: []
tags:
- real_clip
- evaluation
---

## 要約

合成ベンチマーク（[run-i932-synthetic-side-thresholds](000001-run-i932-synthetic-side-thresholds.md)）で決めた閾値のまま、Meiji 3camの全57 clipでsideを判定した。
本番入力の検出器ball（ft-e13）と、評価参照の外注注釈ball（`observed`点だけ）を、同じ`court_side` componentに通した。

- 6 clip（video_000/clip_005、video_001/clip_002・003・006・007・011）は、court検出が`No Court region meets ...`で失敗し、side以前に停止した。
- 残り51 clipで、注釈ballは50 clipで決まり、すべて[F,F,T]だった。margin は0.34以上。停止した1 clip（video_002/clip_001）は、cam0の注釈に`observed`点が無く`disconnected_views`で停止した。clip_000は人手確定の[F,F,T]と一致した。
- 検出器ballは7 clipで決まり（margin 0.15〜0.29）、すべて[F,F,T]だった。注釈と両方決まった6 clipはすべて一致した。誤判定は0。
- 検出器ballの44 clipは理由付きで停止した（`ambiguous_margin` 41、`disconnected_views` 2、`no_consistent_hypothesis` 1）。

## 検出器ballで停止する原因

観測: 検出器ballの最良仮説はcost 0.26〜0.92・support 0.10〜0.75で、注釈ball（cost 0.01〜0.27・support 0.40〜1.00）より大きく悪い。
注釈ballの観測frameに対して、`reprojection_px`（20 px）以内に検出器ballがあるframeの割合（recall）はcam0 21%、cam1 49%、cam2 50%。
検出器が観測したframeのうち注釈ballから20 px以内のもの（precision）はcam0 46%、cam1 63%、cam2 76%。
video_001/clip_008 のcam1やvideo_002/clip_018 のcam1では、検出器がclip全体を通して注釈ballから約460 px離れた別の物体を追っていた（中央値）。

解釈: 停止はside判定の閾値ではなく、検出器ballの誤検出と見落としによる。合成ベンチマークの誤検出はframeごとに独立な乱数だが、実clipの誤検出は同じ物体に長く張り付く系統的なもので、誤った仮説にもある程度整合する。
このため、検出器ballの最良仮説が参照と異なるclipが9あった（最大margin 0.086）。採用したmargin 0.15（合成での誤った最良仮説のmargin最大値は0.103）がそれらをすべて停止させており、marginを下げて判定数を増やすと誤判定が生じる。

## 次に有効な実験

- #934（ball検出器のデータ統一、FT、top-K出力）の後で、同じ`observe`（検出器だけ）と`decide`を再実行する。top-K候補を`court_side`の証拠にできれば、張り付いた誤検出を仮説ごとに選び直せる。
- 実clipに近い誤検出（長く続く静止・準静止の別物体）を合成ベンチマークの条件に加える。
- court検出で停止する6 clipは、court_detection側の課題として別に扱う。
