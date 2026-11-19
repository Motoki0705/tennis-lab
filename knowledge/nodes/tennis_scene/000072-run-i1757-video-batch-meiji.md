---
id: run-i1757-video-batch-meiji
type: run
task: tennis_scene
sequence: 72
recorded_at: '2026-11-19'
title: meiji_4camの3クリップで逐次とモデル再利用を比較する計画
issue: 1757
provider: codex
status: planned
summary_impact: none
parents: [run-i1752-shared-fp16-20261118]
artifacts:
  run_dir: knowledge/runs/run-i1757-video-batch-meiji
---

meiji_4camのclip_000・clip_027・clip_037を固定した。共有court、player証拠のcourt side、ダブルス、DINO・ViTPose fp16、X3D/CSN＋DPT T32の全ball段を、逐次と一括の両方で使う。採用済みのballによるhalf-turnをplayer判定の参照とする。入力・計測条件・GPU probe・集計は[再現手順](../../runs/run-i1757-video-batch-meiji/reproduce.md)にあり、[許容誤差](../../runs/run-i1757-video-batch-meiji/tolerances.json)は実行前に固定した。

先行runの時間・メモリ・保存量を機材の見積りに使った。今回のplayer証拠を含む比較は結果待ちであり、summaryの現在の判断を維持する。
