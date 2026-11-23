---
id: run-i1715-blank-strategy-comparison-20261113
type: run
task: annotation
sequence: 3
recorded_at: '2026-11-13'
title: clip000白紙注釈の4方針比較計画と入力preflight
issue: 1715
provider: codex
date: '2026-11-13'
status: planned
summary_impact: none
config:
  input_gate: blocked_missing_reference
  development_frames: [0, 64]
  cameras: [cam0, cam1, cam2]
  policies: [temporal-overview, full-frame-zoom, sparse-anchor, adaptive]
  planned_camera_runs: 12
  repetitions_per_policy_camera: 1
  agent_model: gpt-6.1-sol
  agent_effort: max
  attempt_seconds: 5400
  selected_policy: null
  later_acceptance_tuned_on_clip000: true
metrics:
  verified_original_videos: 3
  comparison_camera_runs_started: 0
  comparison_camera_runs_completed: 0
  actual_input_tokens: null
  actual_output_tokens: null
  actual_cached_input_tokens: null
  actual_total_tokens: null
repro:
  branch: codex/ball-annotation-clip000-acceptance
  commit: 0ce00c378dcb3a8ed170f481f4e1339ecbd53f6b
  command: bash knowledge/runs/run-i1715-blank-strategy-comparison-20261113/repro.sh
artifacts:
  run_dir: knowledge/runs/run-i1715-blank-strategy-comparison-20261113
  output_dir: outputs/ball_detection/evaluate/blank-strategy-comparison-20261113
parents: []
relations: []
papers: []
tags: []
---

## 結果と残作業

#1715のターン1について、[実行前計画](../../runs/run-i1715-blank-strategy-comparison-20261113/plan.md)と4方針を固定した。3cameraの同じ先頭64frameを各方針で1回ずつ処理する予定で、12 camera run・768判断となる。実GPT比較はまだ一度も起動しておらず、選定方針・unknown・座標誤差・誤不在・実測tokenは未確定である。[未実行の計測表](../../runs/run-i1715-blank-strategy-comparison-20261113/measurements.json)では未測定をnullに保った。3030件の受入workflow、dataset公開、三角測量も未起動である。

[入力preflight](../../runs/run-i1715-blank-strategy-comparison-20261113/preflight.json)は終了値2。指定動画3本は退避されたmultiview-sampleの実ファイルからこのworktreeの`data/samples/`へ復元し、[元の入力記録](../../runs/run-ball3d-evaluation-foundation-20261103/inputs.json)および公式sample検証器と一致した。1010frame・1920×1080・実復号PTSとtime baseも確認した。[復元の来歴](../../runs/run-i1715-blank-strategy-comparison-20261113/input-recovery.json)に取得元と照合結果がある。メインworktreeのファイルは変更していない。

比較の阻害要因は採点用`human_annotations/ground_truth_v1/ground_truth.json`と元CSV6本が現在の実行環境にないことである。固定幾何の`fixed_input_bundle_v1`もないが、これは後の受入runに必要なもので、2D方針比較の追加依存にはしない。[検索した保存先とDriveの結果](../../runs/run-i1715-blank-strategy-comparison-20261113/discovery.json)を残した。検索外の別マシンやバックアップにも存在しないと断定するものではない。元データの復旧が必要であり、検出器予測、旧注釈や別校正から代用正解を作っていない。

## 見込みと解釈の限界

[消費量の計算](../../runs/run-i1715-blank-strategy-comparison-20261113/estimate.json)では、比較/受入の見込み総token比は25.53%で、親指定の50%以下になる。入力tokenの代理はannotation-workflow skillに残る旧予測レビューの概算で、camera runごとの起動量と出力比を明示的な仮定として加えた。白紙注釈の実測ではなく、承認済み資源予算でもない。未知のusageを0にした計算ではない。逐次実行の見込みは比較約101 worker分、受入約353 worker分だが、初回の実測で再見積もりが必要になる。

このprefixは方針調整用で、後の全3030件も「clip000で調整済み」と報告する。1反復・短い先頭区間だけでは後半の遮蔽、打球、連続性の変化や別収録への一般化を比較できない。unknownだけを減らして効率改善とせず、参照が復旧してから応答別usage・可用性・誤不在・位置誤差・画像再閲覧・attemptを同時に比較する。新しい精度合格閾値や削減率目標は設けていない。

`status=planned`は比較が未実行であることを表す。次は元の参照一式の復旧、費用・保存量の実測に基づくruntime/scopeの固定、比較と選定である。受入runは今回の指示では起動せず、選定後の親の次ターン指示を待つ。実装済みCLIの運用・前提不足の記録であり、[summary.mdの注釈に関する現在の判断](../../summary.md#注釈annotationplayer_pose_annotation)は変更しない（`summary_impact=none`）。

## 通常検証

[検証記録](../../runs/run-i1715-blank-strategy-comparison-20261113/verification.json)を参照。新規のproduction実装・テスト追加はない。既存の白紙比較integration 1 nodeを4 pytest workerで実行し、1 passed、全体32.81秒・call 19.92秒・setup 0.02秒であった。実CPUサービスとfake model/synthetic usageによる契約検証であり、GPTの認識精度や実測tokenを証明しない。既存nodeは2方針の入力同一性、台帳の分離、usage欠測、画像再閲覧と参照誤差をまとめて確認するため、重複する新規テストは追加しなかった。
