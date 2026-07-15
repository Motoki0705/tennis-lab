---
id: run-issue-648-multiball-baseline
type: run
title: Multi-ball BLCS track-query baseline (#648)
issue: 648
provider: codex
session: 019f6023-b08a-7af3-aa59-182d7cf43958
date: '2026-07-14'
status: done
config:
  model: track-query-h64-q4-stage2-role-rope
  loss: position+presence; smoothness=0; gravity=0
  data: synthetic-v2; V=2-3; T=8-12; P=1-3; Dmax=6
metrics:
  loss: 0.23678
  loss_position: 0.021762
  loss_presence: 0.215019
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 0.337709
  presence_precision: 0.809143
  presence_recall: 0.890765
  presence_f1: 0.84757
  id_switches: 17.333334
  duplicate_active_tracks: 0.333333
  missed_gt_frames: 9.666667
  inactive_query_false_positives: 9.666667
repro:
  commit: 6ef9adbd7c055a31fdb936da5f475345a3f5d098
  branch: feat/issue-643-648-multi-object-tracking
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: .venv/bin/python -m src.tasks.ball_tracking.scripts.train run.output_dir=outputs/ball_tracking/issue-648-baseline
artifacts:
  run_dir: knowledge/runs/run-issue-648-multiball-baseline
  predictions: knowledge/runs/run-issue-648-multiball-baseline/pred_test.npz
  log: .training_queue/logs/1784025504981160217_1905587_issue-648-multiball-baseline.log
  output_dir: outputs/ball_tracking/issue-648-baseline/logs/version_0
  curves: knowledge/runs/run-issue-648-multiball-baseline/curves.png
  tb_logdir: outputs/ball_tracking/issue-648-baseline/logs/version_0
parents: []
relations: []
tags: [blcs, multi-ball, tracking, mrope, hungarian, synthetic]
---

## 考察 / Findings

### 要約

unordered candidate set から固定 track query を学習する multi-ball BLCS が、smoothness / gravity を使わずに5 epochの学習・validation・testを完走した。testでは `presence_f1=0.847570`、`position_error=0.337709` となり、associationを含む学習基盤が成立した。

### アーキテクチャ詳細

hidden 64、4 query、2 stageの小型baselineである。各stageは `[Q slots + V*D candidates]` のunified self-attention（3軸M-RoPE `(time,camera,role)`）と、`(B*Q,T,H)` のtime-RoPE attentionを交互に適用する。データは `V=2-3`、`T=8-12`、`P=1-3`、`Dmax=6` のon-the-fly syntheticで、dropout、duplicate、random/coherent false positive、camera-time独立shuffleを含む。clip-level Hungarian matching後にposition/presenceを教師し、`loss_smoothness=0.0`、`loss_gravity=0.0` とした。

### メトリクスの解釈

test loss `0.236780` の大半はpresence `0.215019` で、position lossは `0.021762` まで下がった。presenceはprecision `0.809143`、recall `0.890765` でrecall寄りである。`duplicate_active_tracks=0.333333` はslot collapseが限定的である一方、`id_switches=17.333334` とinactive-query false positive `9.666667` はclip内identityがまだ不安定であることを示す。収束曲線ではvalidation lossが約0.34から約0.20へ低下し、途中に一度反発した後、presence loss低下が最終改善を主導した。

### アーキテクチャ⇄メトリクスの因果考察

観測順を座標へ符号化せず、slotだけにlearned identityを持たせたことで、candidate permutation invarianceを保ちながらpresence/locationを学習できた。仮説として、positionは複数cameraの幾何的手掛かりで比較的早く収束する一方、presence中心のmatching costと5 epochの短い学習では交差区間のslot identity維持が弱く、ID switchが残ったと考えられる。

### 既存実験との比較

このIssue専用の最初のmulti-ball runであり、比較可能な親runはまだない。既存single-ball BLCSとは出力契約と評価対象が異なるため数値を直接比較しない。

### 次に有効な実験

同一seed/configで `role_rope_enabled=false` を走らせ、presence F1、ID switch、slot collapseへの寄与を分離する。その後、trajectory crossingとcoherent false-positiveの比率を増やすcurriculum、およびmatching costのpresence weightを下げるablationが有効である。
