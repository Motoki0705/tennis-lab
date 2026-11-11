---
id: group-i1863-review-model-comparison-20261111
type: group
task: review_evaluation
sequence: 15
recorded_at: '2026-11-11'
title: '正式レビュー固定7ケースの親判定: select不採用、Astra固定を維持'
issue: 1863
provider: codex
date: '2026-11-11'
status: done
summary_impact: updated
config:
  paired_repeats: 1
  approved_cases: 7
  model_count: 2
  launch_cap: 16
metrics:
  recorded_model_invocations: 14
  additional_invocations: 0
  parent_adjudication_by_model:
    gpt-6-astra:
      runs: 7
      emitted_findings: 11
      true_positives: 6
      false_negatives: 3
      false_positives: 1
      unconfirmed_findings: 1
      additional_valid_findings: 3
      unconfirmed_finding_rate: 0.09090909090909091
      by_severity:
        blocker:
          expected: 0
          detected: 0
          missed: 0
        major:
          expected: 9
          detected: 6
          missed: 3
      missed_gold_ids:
      - pr-1699-2
      - pr-1867-2
      - pr-1888-2
    gpt-6.1-sol:
      runs: 7
      emitted_findings: 6
      true_positives: 6
      false_negatives: 3
      false_positives: 0
      unconfirmed_findings: 0
      additional_valid_findings: 0
      unconfirmed_finding_rate: 0.0
      by_severity:
        blocker:
          expected: 0
          detected: 0
          missed: 0
        major:
          expected: 9
          detected: 6
          missed: 3
      missed_gold_ids:
      - pr-1699-2
      - pr-1867-2
      - pr-1888-2
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

承認済みの14起動を親が全件判定した。各モデルで正解major 9件中6件に一致し、3件を見逃した。Astraの正解表外の妥当な指摘は3件、誤指摘1件、確認不能1件。Solの正解表外・誤指摘・確認不能は0件。pr-1872-2は両モデルとも契機の異なる緩い一致として親が採用し、その理由を保持した。

条件③は条件①に比べ入力+出力が4.305%、モデル時間が1.621%増え、20%以上改善の基準を満たさないため、親の指示に従いselectを採用しない。既定はfixed（Astra・xhigh）のままとする。1反復だけではSolを通常レビューに使う根拠として不十分であり、反復を増やす推奨案へ進むかは人間の判断待ち。正式レビューの実機受入は親が行う。

summary_impact: updated。[summary](../../summary.md)へ現行方針と未解決点を接続した。全runの使用量は元counterを各1回計上し、投影を物理総量へ足していない。[結果と結論](../../reports/review-model-comparison-20261111/README.md)、[親の全判定](../../reports/review-model-comparison-20261111/adjudications.json)、[数値集約](../../reports/review-model-comparison-20261111/scores.json)を参照。元の再現bundleと各memberは維持している。
