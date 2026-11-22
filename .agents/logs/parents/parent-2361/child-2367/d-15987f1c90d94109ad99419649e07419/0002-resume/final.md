[PR #2372](https://github.com/Motoki0705/tennis-lab/pull/2372) の初回CIで失敗した22件を修正し、すべて再検証しました。同じブランチ `codex/parent-2361-issue-2367-primary` にpush済みです。

- 21件は、watchdogのimportと測定fixtureを修正し、`[local path]` を隠した環境で通過しました。fixtureを外すと、未マウントの理由を表示して終了コード75で拒否することも確認しました。
- 変更前の `main` でも失敗していた手動workerの1件は、一時ミラーを使うよう別コミットで修正し、通過しました。
- 回帰テストは2,028件が通過し、名前空間の権限差で失敗した2件も個別の再実行で通過しました。両コミットのpre-commitも通過しました。

実運用の導入はマージ後です。CIキューの実行中ジョブが完了してから、更新済みのメインcheckoutで実行します。

```bash
sudo systemctl stop tennis-lab-training-queue.service
sudo bash scripts/github_actions/install_self_hosted_runner.sh
spin ops install d-capacity
spin ops on d-capacity
spin ops status d-capacity
cat [local path]
```

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2372,"head":"7fce1cbce63a262d3c1746a8a3d1b72917e18698","resume_on_success":false}}