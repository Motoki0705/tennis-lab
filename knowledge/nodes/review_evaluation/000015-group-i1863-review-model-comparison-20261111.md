---
id: group-i1863-review-model-comparison-20261111
type: group
task: review_evaluation
sequence: 15
recorded_at: '2026-11-11'
title: '正式レビュー固定7ケースのAstraとSol比較: 実行記録、親の判定待ち'
issue: 1863
provider: codex
date: '2026-11-11'
status: done
summary_impact: none
config:
  paired_repeats: 1
  approved_cases: 7
  model_count: 2
  launch_cap: 16
metrics:
  recorded_model_invocations: 14
  additional_invocations: 0
members:
- run-i1863-pr1699-astra-20261111
- run-i1863-pr1699-sol-20261111
- run-i1863-pr1766-astra-20261111
- run-i1863-pr1766-sol-20261111
- run-i1863-pr1867-astra-20261111
- run-i1863-pr1867-sol-20261111
- run-i1863-pr1872-astra-20261111
- run-i1863-pr1872-sol-20261111
- run-i1863-pr1888-astra-20261111
- run-i1863-pr1888-sol-20261111
- run-i1863-pr1686-astra-20261111
- run-i1863-pr1686-sol-20261111
- run-i1863-pr1744-astra-20261111
- run-i1863-pr1744-sol-20261111
papers: []
---

承認済みの固定snapshot比較で14回の実モデル起動を記録した。各runの観測はmembersを正本とし、集計と根拠は[比較レポート](../../reports/review-model-comparison-20261111/README.md)へ接続した。群のdoneは比較実行と記録の完了を表し、指摘の妥当性や採用判断の完了を表さない。

summary_impact: none。既存summaryに本トピックの採用判断は無く、方針変更を行っていない。正解との一致・正解表外の指摘の分類・採否は親が判定する。親は正解指摘を事前に知っていたため、事前固定した選択にも偏りの可能性がある。単一反復、blockerの正解例なし、暫定負例、tool制限、モデル実行順とcacheの影響を残す。正式レビューの実機受入と段階導入は親の残作業である。
