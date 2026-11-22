#2389 を実装し、`codex/parent-2384-issue-2389-primary` の `a614e151e` を push しました。[PR #2398](https://github.com/Motoki0705/tennis-lab/pull/2398)

- `run-worker` は呼出元 checkout のランナー実行環境を予約前と起動直前に検査します。
- `cancel-worker-launch --run-dir` はランナーの終了と run dir の不在を確認し、ログを `not_started` で終結させます。同じ委任で再起動できます。
- `wait-worker --after-run` は、初回または指定した run の次の実行を、記録前から終了・回収まで待てます。

起動・待機の関連114件と運用関連1842件は、再実行を含めてすべて通過しました。Ruff・mypy の pre-commit も通過しています。GitHub CI の確認は親へ引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2398,"head":"a614e151e92d6ea71f6ebee7266b36d3a5d99b42","resume_on_success":false}}