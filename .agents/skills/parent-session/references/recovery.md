# 停止・応答喪失からの復旧

再開時は元のrepo・親番号・stateを使う。GitHubのIssue/PRとローカルの保存は別々に進む。途中失敗なら、どこまで適用されたかを読み、不足した操作から再開する。

| 症状 | 最初に照合するもの | 再開条件と操作 |
|---|---|---|
| 登録・引継ぎ後の表示エラー | `status` の担当と、失敗した操作の応答 `parent_action_applied` | 担当が適用済みなら `display_sync.repair` で表示だけを修復する。 |
| `creation uncertain`、通知の応答喪失 | 親番号・repo、証拠Issueのbinding、通知keyと保存済みspec | `evidence` とGitHubで既存記録を照合する。同一通知が `pending` なら同じspecで続行。送信済みで対象を確認できない場合はjournalを保全し、重複POSTを防ぐ。 |
| 手動ワーカーの結果が返らない | `status` の担当名・委任IDと `inspect-worker` のrun | `wait-worker` の124は待機時間満了。2で `liveness_unknown` / `cleanup_unconfirmed` なら、runnerと子孫の終了・回収を照合する。終了が確認できてから同じ担当を再開する。[runnerの回収手順](../../../ops/lib/README.md#状態確認停止中断からの復旧) |
| 自律親の予約だけ残った | `reconcile` とdispatcherのjob・起動記録 | 委任IDだけが保存されていてjobが未作成なら、そのIDで `dispatcher start-child` を使う。起動済みか不明なら元jobを照合する。[registry欠落時の復旧](../../../ops/dispatcher/README.md#registry欠落からの復旧) |
| 正式レビュー・マージが中断した | `merge-status --run-id RUN`、対象PRの現在のHEAD・merged状態 | 同じrunの `merge-continue` で有効な保存結果と終了を回収する。一時障害なら下記の手順を使う。マージ応答を失ったときはmerge commitを照合して後確認へ進む。切替前にすでにマージ済みのPRには `merge-confirm --prs` を使う。 |
| ログの公開だけが未完了 | `worker-log-status` / `decision-log-status` の対象と理由 | `pending` は同じ記録から公開を続ける。`blocked` は原因を直して再照合。担当終了時の残件は保存された再開コマンドで回収する。[ワーカーログ](../../../ops/worker_logs/README.md#状態と復旧cli)・[判断ログ](../../../ops/decision_logs/README.ja.md#捕捉失敗と復旧) |

### 正式レビューの再試行

`review_usage.rounds` の終了理由と `review.round` を読み、[再試行の条件と上限](../../../ops/review/README.md#正式レビューの障害復旧)を確認する。障害解消後、指示ファイルに続行理由を記し、`merge-continue --run-id RUN --retry-review ROUND --instruction-file recovery.md --wait` を実行する。起動ロックの記録がない旧runでruntimeの結果も見つからない場合は、終了未確認として保留する。前回の作業領域が残っている場合は、ランナーの終了確認後に該当runのworktreeとGit登録を照合し、回収後に同じラウンドを指定する。

次のラウンドの予約後にcontrollerが中断した場合は、通常の `merge-continue --run-id RUN --instruction-file recovery.md` でそのラウンドを継続する。

### ワーカーの引継ぎと版移行

`run-worker` が共通ランナーの実行ファイルを理由に起動前拒否した場合は、呼出元checkoutの `.venv/bin/python` とランナーを整備して再実行する。

`runtime_unconfirmed` の手動起動は、`run-worker` または `inspect-worker` が返した `run_dir` を `cancel-worker-launch --child CHILD --run-dir RUN_DIR` で指定できる。複数担当なら `--worker` も指定する。最新の予約を照合し、ランナーの消滅とrun dirの不在を確認できた場合に、ログ捕捉を `worker_failed / not_started` として終結して起動予約を取り消す。その後は同じ委任で元の `run-worker` を再実行する。消滅を確認できない場合やrun dirが存在する場合は、runnerの終了と回収の照合へ進む。

正式レビューの修正ターンも確認済みの未起動予約を取り消せる。同じ修正ID・PR・指示で元のrunから再開すると、新しいrun dirを割り当てる。取消済みのログは `worker-log-status` で確認できる。

通常の修正は同じワーカーのsession・branch・worktreeで継続する。手動委任で元sessionが使えないと保存記録から確かめた場合に限り、`run-worker --replace-session --prompt-file next.md --restart-evidence-file reason.md` で次の指示と、終了・再開不能の証拠を渡す。

旧Astra担当に修正が残るときは、終了・回収を確認し、元のPR・branch・worktree、残るmajorと検証結果を引き継ぐ。手動は `delegate --handoff-from-astra 旧委任ID --restart-evidence-file 引継ぎ文` のあと通常の `run-worker`、自律は `start-worker --handoff-from-astra 旧委任ID --handoff-evidence-file 引継ぎ文`。各コマンドには通常の担当・worktree等の引数も渡す。

CLI更新後は新しいcheckoutから同じstateを使う。version 1のregistryを移す場合は全親の呼出元を揃えてから `state-migrate`。退避した `registry-v1.backup.json` には移行後の更新が含まれないため、復旧時は現在の親別記録と照合する。
