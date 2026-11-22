#2367を実装し、[PR #2372](https://github.com/Motoki0705/tennis-lab/pull/2372) を作成しました。ブランチは `codex/parent-2361-issue-2367-primary`、HEADは `3afef2f42739294374b9b67cb8e6d63b59171c15` です。

物理D:の空きが既定の20 GiBを下回ると、ローカル・CIキュー、dispatcher、`agent_exec.sh` の新規起動を保留します。実行中のジョブは継続します。watchdogは30秒ごとに測定し、CIキューは判定が欠落している場合や90秒を超えた場合も起動を保留します。

- 実機の空き366.23 GiBに対して閾値を一時的に367 GiBに設定し、ローカルキュー・dispatcher・`agent_exec.sh` の拒否を確認しました。キューは閾値を20 GiBに戻すと同じジョブを完了しました。
- Windows通知履歴でトースト1件を確認し、同じ状態の再測定でも1件でした。回復時の通知は単体テストで確認しました。
- 追加16件とキュー関連175件が通過しました。運用関連は並列で758件が通過し、タイムアウトした1件も単独で通過しました。pre-commitも通過しました。

実運用の導入とCIサービス内の照合はマージ後です。CIキューの実行中ジョブが完了してから、更新済みのメインcheckoutで実行します。

```bash
sudo systemctl stop tennis-lab-training-queue.service
sudo bash scripts/github_actions/install_self_hosted_runner.sh
spin ops install d-capacity
spin ops on d-capacity
spin ops status d-capacity
cat [local path]
```

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2372,"head":"3afef2f42739294374b9b67cb8e6d63b59171c15","resume_on_success":false}}