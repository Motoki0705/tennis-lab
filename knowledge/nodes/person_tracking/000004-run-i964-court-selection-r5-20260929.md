---
id: run-i964-court-selection-r5-20260929
type: run
task: person_tracking
sequence: 4
recorded_at: '2026-09-29'
title: 隣コート混入の原因分離と断片連結後の滞在選別
provider: codex
status: done
config:
  device: cpu
  split: four existing Meiji dev clips
  selection:
    ambiguity_margin: 0.2
    ambiguous_min_s: 1.0
    baseline_margin_m: 5.0
    max_candidates: 6
    max_gap_s: 1.0
    max_handoff_s: 0.2
    max_speed_m_s: 6.0
    min_cosine: 0.8
    min_presence_fraction: 0.25
    position_slack_m: 0.8
  appearance: saved run-4 CLIP only; missing explicit; pipeline default unchanged
metrics:
  player_units: 20558
  adjacent_units: 395
  ft_kept_player_units: 18116
  ft_cam0_far_kept: 2521
  ft_rejected_adjacent: 395
  union_kept_player_units: 19320
  union_cam0_far_kept: 3072
  union_rejected_adjacent: 395
  old_pipeline_kept_player_units: 19444
  review_video_frames: 320
artifacts:
  run_dir: knowledge/runs/run-i964-court-selection-r5-20260929
  output_dir: /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/court_selection/i964-cpu-r5-final-20260929
parents:
- run-i964-court-selection-cpu-r4-20260929
relations: []
papers: []
tags: []
issue: 964
date: '2026-09-29'
repro:
  commit: b9fec834949ca002ff15804dd3b6f0839a1b0745
  branch: campaign930/i964-2-tracking
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
    PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i964-2-tracking
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i964-2-tracking/.venv/bin/python
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i964-2-tracking/tests/benchmarks/person_selection_refinement.py
    --previous /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/court_selection/i964-cpu-r4-20260929
    --report /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/court_selection/i964-cpu-r5-final-20260929
---

## 結論

隣コートの誤採用は、確認できるラベルunitでは選手とのidentity mixingではなく、主コートの横余白が隣人物の足元を含むためだった。FTの395 unitとunionの389 unitはいずれも隣コート人物だけのtrackに属する。単に横余白を0にしても97 unitがダブルス幅内に投影されるため、滞在coreと外側境界を分けた。

校正足元で断片を連結して滞在を合算する改善版では、全3保存ソースで隣コート395/395 unitを除外し、FT/unionのcam0 far保持が改善した。全8選手label-IDを50%保持基準で残す。一方、他cameraの遠側と旧経路の選手保持には損失があり、FTのコート外unit除外は悪化する。既定採用は見送る。全表、ルール、原因、失敗した試行と動画は[run 5レポート](../../runs/run-i964-court-selection-r5-20260929/report.md)を正とする。

## 解釈と限界

regionだけの変更に連結を足すablationで改善を確認した。ただしraw trackと初回採用trackのCLIPだけを再利用しており、全人物の外観はない。欠測を隠さず、次段階で補う必要がある。画像下端の切れ・校正誤差・box下端のノイズによる位置誤差の保証はなく、横に走り出た選手も削る。4 singles dev clipでの調整をダブルスや未見の性能とみなさない。

初回は単frame跳び検査で過分割になった。既存の時間窓検査へ戻し、gap 0.5/1/2秒とambiguity marginをdevで測り、1秒/0.2を暫定採用した。小さいmarginはコート外人物も戻すため採用しなかった。失敗表・logもbundleへ保存し、最良行だけを記録していない。

COCO/unionはまだROI後保存との比較で不公平なため、全画面COCOを1件だけqueueへ投入した。GPU jobと見積りはレポートに記録し、その結果と公平な最終3ソース比較は次run。旧box一致はCOCO由来なので検出recallと呼ばない。今回のCPU選別は最終追跡方式比較やcamera間対応の再評価ではない。

## 次の実験

全画面COCO archiveの完了/hashを回収し、ROI前の同一条件でFT/COCO/unionを比較する。閾値sweepはCPU。group出力をperson_identities v3の安全策に接続し、第2確認で残る非選手を測る。2–3追跡方式と両baseline、CLIP-ReID/SOLIDER/KPR encoder、全pipeline、調整凍結後一度だけの未見評価は未完了。TensorBoardは推論/CPU解析のみのため対象外。
