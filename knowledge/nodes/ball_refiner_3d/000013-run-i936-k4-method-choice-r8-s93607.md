---
id: run-i936-k4-method-choice-r8-s93607
type: run
task: ball_refiner_3d
sequence: 13
recorded_at: '2026-09-30'
date: '2026-09-30'
title: 固定K=4標本でのNLL・HDR・CPU費用による方式選定
provider: codex
status: running
issue: 936
config: {sample_seed: 93607, frames: 311, components: 125, workers: 4, hdr_samples: 8192}
metrics: {}
artifacts: {run_dir: knowledge/runs/run-i936-k4-method-choice-r8-s93607}
parents: [run-i936-k4-audit-r7-s93607, run-i936-provisional-degradation-r5-s936]
relations: []
papers: []
tags: [cpu, probabilistic-triangulation]
---

run 8 directiveに従い、方式選定を仕様のGT 3D NLL・HDR coverage・計算費用へ戻す。
run 7の数値収束目標は診断として残し、採用や生成のgateにはしない。
事前登録311frameと#959暫定入力は固定する。GTは採点だけで使用する。

比較source/予算は7392f7f6で結果採点前にcommit。Aは正depth最適化の最終点と
Gauss–Newton covarianceを全成分について返し、非正則理由を保存する明示的な近似。
Hは8³/4段/64cell/prior±5σのvoxelを使う既存hybrid、ray/adaptiveはrun 7と同じ予算、
Cは全成分積それぞれ8サンプル・seed93608の摂動最適化、Q20は固定20次ray。
全125成分とcamera subsetのBernoulli質量を保つ。4 worker/native thread1。

初期A測定でfloat32保存のSPD検査が100frameの採点を止めた。
float64分布のNLL/HDR評価と保存の可否を分けるため、採点器だけを修正して
同じ全311frameを再測定した。方式・予算・入力・seed・閾値は変えていない。
初期結果はA-export-gate-results.json.gzに残す。float32不適合を修復・隠蔽せず診断に残す。

採点は自然対数のGMM密度(m⁻³)、GTと混合平均のL2距離(m)。HDRは各frameの
混合分布から8192標本を引き、log qの(1−α)分位点以上の密度集合と定義する。
seedは936080000+事前登録frame indexで全方式共通。50/90/95%を報告する。
採点やI/Oを含まないsolver wall秒/frameと、実行全体のwallを区別する。
停止データ3995frameの層比率加重と標本全体を分け、処理失敗は省略せず母数と併記する。

40件の関連テスト、ruff/mypyが成功。前回38843ed4のPython CIは6,556 passed、
133 skippedで成功（bundleのCI JSON）。6候補の全標本測定を実施中で、方式選定は未確定。
96/640件の生成、GPU、実Meiji球評価、pipeline接続は実施しない。
