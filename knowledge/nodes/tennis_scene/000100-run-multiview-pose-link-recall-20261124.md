---
id: run-multiview-pose-link-recall-20261124
type: run
task: tennis_scene
sequence: 100
recorded_at: '2026-11-24'
title: 関節マスクと胴体の対応スコアでMeijiの正解リンクを回復
issue: 2444
provider: codex
date: '2026-11-24'
status: done
config:
  device: cpu
  assignment: paired torso geometry and exclusive MILP
  joint_quality: masked joint and depth-conditioned proximal limb checks
  meiji4_workers: 4
metrics:
  dev_micro_pair_f1: 0.8514540781820641
  dev_false_pairs: 1045
  dev_missed_true_pairs: 4446
  previous_lost_true_links: 2458
  previous_gained_true_links: 980
  lovo_p90_px: 43.64250415228867
  lovo_scored_joints: 267019
  meiji4_clips: 69
  meiji4_source_frames: 46228
  meiji4_populated_frames: 46228
  meiji4_assignment_rate: 0.8589779372374955
  meiji4_wall_seconds: 367.6801645469968
  unseen_score_attempts: 1
  unseen_all_three_micro_pair_f1: 0.7516561568056093
  unseen_without_repaired_micro_pair_f1: 0.740628047591184
  unseen_all_three_player_micro_pair_f1: 0.862066264588392
  unseen_without_repaired_player_micro_pair_f1: 0.8702786945361203
artifacts:
  run_dir: knowledge/runs/run-multiview-pose-link-recall-20261124
  meiji3: knowledge/runs/run-multiview-pose-link-recall-20261124/meiji3.json
  meiji4: knowledge/runs/run-multiview-pose-link-recall-20261124/meiji4.json
  losses: knowledge/runs/run-multiview-pose-link-recall-20261124/losses.json
  holdout_inputs: knowledge/runs/run-multiview-pose-link-recall-20261124/holdout-inputs.json
  holdout_camera_recovery: knowledge/runs/run-multiview-pose-link-recall-20261124/holdout-camera-recovery.json
  holdout_pose_preparation: knowledge/runs/run-multiview-pose-link-recall-20261124/holdout-pose-preparation.json
  holdout_pair_score: knowledge/runs/run-multiview-pose-link-recall-20261124/holdout-pair-score.json
parents:
- run-multiview-pose-meiji-20261124
- run-i964-unseen-r16-20261001
- run-slcs-meiji-court-visual-qc-v1
relations: []
papers: []
tags:
- meiji
- multiview
- cpu
- frame_pose
summary_impact: none
repro:
  commit: 590c2377810762fdea0d5485d10307e612de51eb
  branch: codex/parent-1739-issue-2444-primary
  command: bash knowledge/runs/run-multiview-pose-link-recall-20261124/reproduce.sh holdout-score "$replay_dir"
---

Meiji 3camの開発4 clipでは、初回全身版の取りこぼし増分1,478件が、2,458件の消失と980件の新規採用から生じた。骨長で1,280件、MILP競合で1,142件失った。肩・腰のスコアと全身の品質判定を分けると、micro pair F1は0.8050から0.8515、FPは906から1,045件になった。試作のF1 0.8485とFP上限1,102件を満たした。未使用視点で採点できる関節は増え、P90再投影誤差は27.81 pxから43.64 pxに上がった。

meiji_4camは全69 clip・46,228フレームで候補を生成し、85.90%のDINO行を割り当てた。4 CPUプロセスで367.68秒。標準パイプラインの採用判断は据え置く。 [設定選択と実データ結果](../../runs/run-multiview-pose-link-recall-20261124/README.md)

同じ映像の過去のCourt投影点を使ったcam0の復元はRMSE 3.13 pxで校正上限を通過した。r18のcam1・cam2と組み合わせると、同一clipのボール注釈でhalf-turnが `[false, false, true]`、次点との差が0.475となった。親が承認した補完を使い、3 clip・9視点の38,421行のViTPoseを固定計画で取得した。

予約未見をr18の人手部分参照で一度だけ採点した。micro pair F1は全3 clipで0.7517、`video_001/clip_003` を除く2 clipで0.7406。選手同士では0.8621と0.8703だった。38,132のラベルboxがすべて検出行に対応した。採点版と固定された計画・ViTPose・参照ラベルを使って再現する。 [固定版の手順](../../runs/run-multiview-pose-link-recall-20261124/README.md#未見評価の再現) [採点結果と入力SHA](../../runs/run-multiview-pose-link-recall-20261124/holdout-pair-score.json)
