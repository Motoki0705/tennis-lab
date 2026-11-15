[PR #2087](https://github.com/Motoki0705/tennis-lab/pull/2087) を作成しました。`Closes #2080` 記載済みです。

- branch: `codex/unassigned-decision-publication`
- 最終HEAD: `79b6b797834f225f9b86fc7993e11d5aca0abe60`
- 関係テスト702件、最終修正後の終了・回収93件が成功。pre-commit成功、テストプロセス残存なし。
- [最新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43659299676) は、Pythonを含む14ジョブすべて成功しました。

操作sessionが回収親をactiveで担当し、対象がregistry行なし・ended・releasedの場合に受け付けます。対象がactive／handoff待ち、操作側が非active・別session、担当・世代の変更、registry破損時は拒否・中断します。対象の担当・世代・registryを変えず、公開担当を記録します。

merge後、#1900の担当sessionで実行できます。先に末尾へ `--dry-run` を付けて確認してください。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo Motoki0705/tennis-lab --parent 1900 publish-decision-log --unassigned-parent 1796
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo Motoki0705/tennis-lab --parent 1900 publish-decision-log --unassigned-parent 1864
```

追加テストは計27ケースです。

- `test_unassigned_parent_cli_publishes_open_and_closed_without_assignment`
- `test_unassigned_parent_cli_refuses_ownership_before_any_mutation`
- `test_unassigned_publication_rechecks_before_external_writes`
- `test_unassigned_publication_reuses_existing_batch_and_records_recovery_actor`
- `test_publication_actor_must_reference_its_frozen_request`

実stateは読み取りとdry-runのみ実施しました。残作業は親による統合レビュー・merge・実公開・main表示確認です。広域opsの任意mypyでは既存16件を確認し、変更前との差分は0件でした。validatorは未指定のため試行0／完了0です。