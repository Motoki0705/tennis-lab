# 停止・応答喪失からの復旧

| 症状 | 確認と次の操作 |
|---|---|
| 担当表示だけが失敗 | `status` の担当を照合し、適用済みなら `display_sync.repair` の表示修復を行う。 |
| 証拠Issue作成・通知の応答を失った | `evidence` または通知keyで既存の記録を照合する。作成の成否が不明な間は同じPOSTを重ねない。 |
| ワーカーの起動・終了が不明 | `status` → 対象の `inspect-worker`。委任ID・branch・worktree・runを固定して終了とcleanupを確認する。`liveness_unknown` や `cleanup_unconfirmed` なら [runnerの回収手順](../../../ops/lib/README.md#状態確認停止中断からの復旧) で照合し、その終了証拠で再予約する。旧Astraからの明示引継ぎは `delegate --help` または `start-worker --help`。 |
| 自律親の委任だけ予約済み | 保存済み委任IDに `dispatcher start-child` を接続する。registryが欠落している場合は [dispatcherの復旧](../../../ops/dispatcher/README.md#registry欠落からの復旧) を使う。 |
| レビュー・マージの応答を失った | `merge-status --run-id` で保存段階を読み、同じrunを `merge-continue`。保存結果と回収を確認できないPRは親へ返る。切替前にマージ済みのPRは `merge-confirm --prs` で後確認する。 |
| ログの公開だけが止まった | `worker-log-status` / `decision-log-status` で対象を特定し、同じ記録から公開を再開する。公開結果と作業の成否は別々に扱う。[ワーカーログ](../../../ops/worker_logs/README.md#状態と復旧cli)・[判断ログ](../../../ops/decision_logs/README.ja.md#捕捉失敗と復旧) |

CLI更新後は新しいcheckoutから同じstateを使う。version 1のregistryを移す場合は全親の呼出元を揃えてから `state-migrate`。退避した旧版には移行後の更新が含まれない。
