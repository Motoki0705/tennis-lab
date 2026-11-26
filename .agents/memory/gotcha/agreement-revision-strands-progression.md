---
title: 合意の改訂は進行中のレビュー・マージ後確認を止める
summary: 親の合意をagreeで改訂すると、改訂前に始めたレビュー進行は権限変更で続行できず、同じPRのmerge-confirmも拒否される。改訂前にマージ後確認まで終える。
applies_to: all
source: '2026-11-26 #1722 で合意を改訂3に更新した後、#2571 のマージ後確認が続行できなかった（親セッションの観察）'
created: &id001 2026-11-26
last_verified: *id001
evidence: post_merge_wait のrunがある状態で agree を保存し、merge-continue と merge-confirm --prs の応答を確かめる。
citations:
- path: .agents/ops/review/merge_gate_impl.py
  symbol: progression execution authority changed
- path: .agents/ops/review/review_locks.py
  symbol: PR has an unfinished progression; continue that request
---

- 進行runは開始時の `authorization`（実行根拠のSHA-256を含む）を保存し、続行時に現在の値との一致を求める。`agree` で目的・範囲を足しただけでも一致しなくなり、`merge-continue` は「progression execution authority changed」で止まる。
- そのPRは古いrunが未完了のまま保持するため、`merge-confirm --prs` で新しい確認runを作ろうとしても「PR has an unfinished progression」で拒否される。
- 合意を改訂する前に、`merge-status` で全PRが `completed` になったことを確かめる。main CIが同じconcurrency groupの後続pushで `cancelled` になった場合は、改訂前に `gh run rerun` で確認を済ませる。
- すでに止まった場合も、状態ファイルを手で書き換えない。PRのマージ・merge commitのCI・Issueの状態を照合して記録に残す。
