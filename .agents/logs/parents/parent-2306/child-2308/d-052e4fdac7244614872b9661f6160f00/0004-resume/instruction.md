# PR #2313 追加修正: CIで見つかった退行

旧HEAD 11c1396ce の GitHub CI（run 44671284800, `CPU tests (0)`）で次の1件が失敗した。親が新HEAD 3a49c69a6 でもローカルで再現し、origin/main（f6751b866）では通ることを確認した。このPRによる退行。

```
FAILED tests/e2e/agents_ops/test_decision_capture.py::test_failures_never_become_success_or_parent_acceptance[final_missing]
AssertionError: assert 'success' in {'failure', 'partial', 'rejected', 'unknown'}
```

最終報告が空（`MANUAL_PLANNING_FINAL=""`）のrunを、`parent.run-worker` の判断記録が `success` として残している。最終報告の欠落を成功扱いにしてはいけない（静かなフォールバック禁止）。PARENT_RESULT 宣言の検査や worker_wait の変更が、欠落時の分類を変えていないか確認し、欠落は従来どおり失敗系の状態になるよう直す。宣言の欠落（自由文はあるが PARENT_RESULT 行がない）と最終報告そのものの欠落の扱いが矛盾しないことも確かめる。

- 修正後、`tests/e2e/agents_ops/test_decision_capture.py` 全体と、前回の回帰テスト・関連e2e・pre-commitをローカルで通す。他にも main と比べて落ちるテストがないか、`tests/e2e/agents_ops/` の CPU テストを広めに実行して確認する。
- commit・push して終了する。CIを待たない（sleep・照会ループ・gh run watch・gh pr checks --watch 禁止）。
- 最終報告に原因・修正・実行したテストと結果・新しい完全HEADを書く。非公開パス・セッションIDを書かない。作業時間の上限は2時間。
