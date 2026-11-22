正式レビューの4件を修正し、`codex/parent-2361-issue-2367-primary` の `4739c50aa` をpushしました。検証結果と導入手順を追記しました。[PR #2372](https://github.com/Motoki0705/tennis-lab/pull/2372)

- 共通ランナーがCLIを起動する前に容量を理由に拒否した場合、dispatcherは同じ子予約に戻し、手動委任は同じ委任とセッションから再試行できます。
- 正式レビューはCLIの未起動を確認できた場合に起動枠を戻します。`merge-continue` で同じレビュー回が完了し、CLI起動は1回でした。
- Windows通知の履歴を永続化しました。未送信が確定し、同じ判定が続く場合は次の測定で再試行します。送信結果が不明な場合は重複を抑止し、判定が変わった場合は新しい状態を通知します。
- CI用キューは `CLOCK_BOOTTIME` とboot IDで判定の鮮度を確認します。壁時計の巻き戻しや再起動で期限切れの許可が復活しないことを確認しました。

修正前に9件の失敗を再現しました。修正後は対象23件、回帰2,477件、pre-commitが通過しました。変更後の通知スクリプトは実機のWindows PowerShellで構文解析を通過しました。通知の実送信と履歴は初回実装時に確認済みです。

マージ後は、実行中のCI用キュージョブが完了してから、更新したメインcheckoutで導入と状態確認を行います。

```bash
sudo systemctl stop tennis-lab-training-queue.service
sudo bash scripts/github_actions/install_self_hosted_runner.sh
spin ops install d-capacity
spin ops on d-capacity
spin ops status d-capacity
cat [local path]
```

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2372,"head":"4739c50aad50978c28d1ff603f830807f6ea68f5","resume_on_success":false}}