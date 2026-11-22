Python CIの失敗を修正し、`codex/parent-2361-issue-2367-primary` に `f7dbdd733` をpushしました。検証結果を追記しました。[PR #2372](https://github.com/Motoki0705/tennis-lab/pull/2372)

容量拒否の照合をランナー側へ移し、AGYの単独CLIとhookが `.agents/ops/lib` だけでイベント解析を読み込めるようにしました。タイムアウトしたeditorテストには、起動通知を待つ15秒の期限と、失敗時の取消処理を加えました。

CIで失敗・未完了だった97件はすべてローカルで通過しました。うちブラウザが必要な2件は、CIと同じPlaywright版で別途検証しました。容量判定と再試行を含む回帰386件、pre-commit、`git diff --check` も通過しました。

残りは新HEADのCI確認と、マージ後の導入です。導入は実行中のCI用キュージョブが完了してから、更新したメインcheckoutで行います。

```bash
sudo systemctl stop tennis-lab-training-queue.service
sudo bash scripts/github_actions/install_self_hosted_runner.sh
spin ops install d-capacity
spin ops on d-capacity
spin ops status d-capacity
cat [local path]
```

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2372,"head":"f7dbdd733d1370358abe9b7b1ba1ecf6f748dcee","resume_on_success":false}}