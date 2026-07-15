---
id: run-issue-643-multiperson-baseline
type: run
title: Multi-person PLCS track-query baseline (#643)
issue: 643
provider: codex
session: 019f6023-b08a-7af3-aa59-182d7cf43958
date: '2026-07-14'
status: done
config:
  model: track-query-h64-q4-stage2-role-rope
  loss: position+rotation+presence; track_smoothness=0
  data: synthetic-v2; V=2-3; T=8-12; P=2-3; Dmax=5
metrics:
  loss: 0.618088
  loss_position: 0.014382
  loss_rotation: 0.322732
  loss_presence: 0.44234
  loss_track_smoothness: 0.0
  position_error: 0.267685
  angular_error_deg: 41.522079
  presence_precision: 0.67034
  presence_recall: 0.885713
  presence_f1: 0.762981
  id_switches: 23.666666
  duplicate_active_tracks: 6.333333
  missed_gt_frames: 14.666667
  inactive_query_false_positives: 39.666668
repro:
  commit: e0cf7fc5d24cc28fa6ecfaaae94a8da2642f5469
  branch: feat/issue-643-648-multi-object-tracking
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: .venv/bin/python -m src.tasks.player_tracking.scripts.train run.output_dir=outputs/player_tracking/issue-643-baseline
artifacts:
  run_dir: knowledge/runs/run-issue-643-multiperson-baseline
  predictions: knowledge/runs/run-issue-643-multiperson-baseline/pred_test.npz
  log: .training_queue/logs/1784025505000778325_1905602_issue-643-multiperson-baseline.log
  output_dir: outputs/player_tracking/issue-643-baseline/logs/version_0
  curves: knowledge/runs/run-issue-643-multiperson-baseline/curves.png
  tb_logdir: outputs/player_tracking/issue-643-baseline/logs/version_0
parents: []
relations: []
tags: [plcs, multi-person, tracking, mrope, hungarian, synthetic]
---

## 考察 / Findings

### 要約

unordered multi-view pose detectionsから固定queryでclip-local player trackを学習するbaselineが、track smoothnessなしで5 epochの学習・validation・testを完走した。testは `position_error=0.267685`、`angular_error_deg=41.522079`、`presence_f1=0.762981` で、multi-person PLCSの学習経路とtracking診断が成立した。

### アーキテクチャ詳細

hidden 64、4 query、2 stageの小型baselineである。各person detectionを1 tokenへ変換し、`[Q slots + V*D player tokens]` に3軸M-RoPE `(time,camera,role)` を適用したspatial self-attentionと、slotごとのtime-RoPE attentionを交互に行う。synthetic dataは `V=2-3`、`T=8-12`、`P=2-3`、`Dmax=5` で、articulated motion、birth/death、joint/detection dropout、false positive、camera-time独立shuffleを含む。Hungarian matching後にposition/rotation/presenceを教師し、Issue方針どおり `loss_track_smoothness=0.0` とした。

### メトリクスの解釈

test loss `0.618088` の内訳はposition `0.014382`、rotation `0.322732`、presence `0.442340` で、positionよりorientationとpresenceが支配的である。presenceはrecall `0.885713` に対しprecision `0.670340` で過検出傾向があり、inactive-query false positive `39.666668`、duplicate active tracks `6.333333` に表れている。validation lossは約0.76から約0.59へ一貫して低下し、positionは約0.041から約0.018へ改善した。一方、角度誤差は約55°から約42°へ早期改善後に頭打ちとなった。

### アーキテクチャ⇄メトリクスの因果考察

camera-time内のdetection indexを一切符号化しない構造でも、positionとpresenceに有効な観測集合をslotへ集約できた。仮説として、2D skeletonの左右・camera方位とyawの対応が短いsynthetic runでは十分に学習されず、rotation headが早期にplateauした。またno-person queryへのpresence supervisionは入るが、position/rotation supervisionはmatched slotだけなので、unused slot間の競合が弱くduplicateとfalse positiveが残ったと考えられる。

### 既存実験との比較

このIssue専用の最初のmulti-person runで、比較可能な親runはない。既存single-person PLCSとはtrack associationとpresenceを含む契約が異なるため、position/rotation値を直接baseline比較には使わない。

### 次に有効な実験

まず同一seed/configのrole RoPE OFF ablationを行う。次にunmatched queryへのpresence weight増加、query間repulsion/duplicate penalty、長いclipでの交差場面比率増加を順に検証する。rotationについてはcamera contextをdetection encoderへ明示的に与えるablationと、学習epoch延長を分けて試す。
