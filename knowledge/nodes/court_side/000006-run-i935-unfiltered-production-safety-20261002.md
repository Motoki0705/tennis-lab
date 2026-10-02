---
id: run-i935-unfiltered-production-safety-20261002
type: run
task: court_side
sequence: 6
recorded_at: '2026-10-02'
title: confidenceフィルタ廃止のCPU回帰と安全benchの元入力欠測
issue: 935
provider: codex
date: '2026-10-02'
status: failed
config:
  requested_conditions: 28
  requested_scenes: 400
  seed: 1
  residual_seed: 30001
  side_margin: 0.15
metrics:
  safety_cases_executed: 0
artifacts:
  run_dir: knowledge/runs/run-i935-unfiltered-production-safety-20261002
parents:
- run-i935-correlated-safety-r30-20261001
relations: []
papers: []
tags: []
repro:
  commit: 37c41ce42c5e5b79110b07cee97088795b246f25
  branch: codex/remove-ball-confidence-filter-20261002
---

## 判断と実装

2026-10-02のユーザー指示「このフィルタを廃止します。実装から取り除いて元のフィルタなしの方法に戻してください」に従い、
#973のproductionから存在確率・90%包含面積による点棄却を廃止した。
e9＋anchored_12k seed42＋共分散倍率1.8125148752087792、court_sideのball-only/margin .15は保持する。
同日の追加回答「既存の無選別結果・回帰テスト・最新CIを根拠に進め、再測定未実施を明記する」により、
元datasetを再生成せず、従来stackのmergeを進める。採用・merge方針の変更であり、
過去の精度判定を新しい合格観測に変更する意味ではない。
保存・下流・旧artifactの扱いは[点consumer契約](../../../src/tennis_scene/pipeline/README.md#refinerの点consumer)を正本とする。

CPU回帰では、存在確率が0に丸められ、大きい共分散を持つframeも最大weight成分の平均点として下流へ届くこと、
全GMMのtensor・camera・実PTS・窓出自をreadbackできること、v2のexecute/load、旧v1の拒否、旧gallery閲覧を検証した。
旧未選別座標とのbit一致も検査した。閾値helperと旧規則は過去bench再現専用としてtests/benchmarksに隔離し、
productionのconfig/APIから参照しない。検出器の点へのfallbackは追加していない。

## 安全benchは元入力がなく未実施

[run30](000005-run-i935-correlated-safety-r30-20261001.md)と同じsplit・seed・28条件×400sceneを
productionの`BallPointsModule`へ通すCPU入口を追加した。実行時、元dataset
`data/blcs/single_object_camera_view_v2/test.txt`が存在せず停止し、**scene-conditionを1件も測定していない**。
このノードの`failed`は入力欠測による実行失敗であり、新しいwrong decisionの観測を意味しない。
固定残差bankと較正artifactのSHA検査までは通過した。別datasetへの置換・再生成は行わない。
[実行log](../../runs/run-i935-unfiltered-production-safety-20261002/bench.log)、
[状態と旧結果への参照](../../runs/run-i935-unfiltered-production-safety-20261002/result-status.json)、
[再現入口](../../runs/run-i935-unfiltered-production-safety-20261002/repro.sh)を保存した。
先行するmodule入口の環境衝突は[別log](../../runs/run-i935-unfiltered-production-safety-20261002/invocation-failed.log)に残し、
実行入口を既存benchと同じ直接script形式へ修正した。

旧run30の未選別wrong 0/11,200・停止35.3482%、選別wrong 2/11,200・停止28.4286%は過去の結果として保持する。
今回の実装で全条件のwrong 0を再測定できたとは主張しない。元の合成摂動＋実残差移植という外的妥当性の限界も同じ。
旧run29/30のFAIL、元動画B gate FAIL、seed再現FAILは修正しない。
clip_00019/28などのcourt_side停止、clip_000全scene、人物pair F1 .719701/2-of-3の再評価は未実施。
GPU推論・学習は行っておらず、TensorBoard/学習曲線は対象外。独立validator指定・試行・完了は0回。
