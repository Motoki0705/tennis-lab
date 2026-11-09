pr-1766-1を修正し、同じ[PR #1766](https://github.com/Motoki0705/tennis-lab/pull/1766)へpushしました。未mergeです。

- 修正commit: `5f0931a04`
- 自動許可を同一attemptの`timed_out`に限定。保存判断の再適用時も取消状態・理由を保持します。
- 追加テスト: `test_cancellation_after_automatic_decision_requires_explicit_retry`。修正前の不具合を再現し、修正後は成功。
- 対象31件成功（68.47秒）。pre-commitのruff・mypyも成功。
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42528085228)は、`Python`を含む全13 jobs成功。

証拠はPRと#1709へ記録済みです。親の指摘解消確認・mergeへ返します。