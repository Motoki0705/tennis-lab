# PR #2313 正式レビュー（major）の修正ターン

正式レビュー run `parent-2306-4fcf9412d4a54f04809445f9df2fa4a9`（対象HEAD 11c1396ce）が major 6件を返した。親は6件すべてを採用する。各指摘を修正し、それぞれに回帰テスト（指摘が示す順序・競合を再現するもの）を加える。

## 指摘（原文）

### 1. 別PRのCI監視でレビューの部分再開が無通知のまま止まる
静的な推論。複数PRの進行でAだけを `merge-continue --prs A --wait` すると、`continue_progression` はAの後続処理だけを検査します。一方、起動後の `review_locks.targets` は全PRのロックを取得します。Bの手動監視がロックを保持していると、進行状態を `running` に保存した後でコントローラーがstderrだけを残して終了し、親への通知が出ず `--wait` が戻りません。事前検査と実際のロック範囲を揃え、起動後の競合も親が観測できる状態にしてください。新規テストは手動監視側がロックを取れない場合のみを検証しています。 [merge_gate_impl.py](https://github.com/Motoki0705/tennis-lab/blob/11c1396cebe06885c9489626990be82a592c908e/.agents/ops/review/merge_gate_impl.py#L316) [review_locks.py](https://github.com/Motoki0705/tennis-lab/blob/11c1396cebe06885c9489626990be82a592c908e/.agents/ops/review/review_locks.py#L30)

### 2. 別ワーカーが空き枠を使うと自動再開が打ち切られる
静的な推論。`run_worker` は待機宣言を受け付ける前に `worker_ended` を記録します。委任上限1でAがキュー待ちに入ると、`launch_plan` は枠が空いたと判定し、親はBを正常に起動できます。Bの実行中にAのジョブが終わると、監視は再開イベントを消費した後、枠上限でAの再開を拒否されて終了します。Bが終了してもAは自動再開されません。枠を保持するか、空き枠を再取得できるまで再開イベントを保持する必要があります。既存の枠再取得テストに、この2担当の順序を加えると検出できます。 [manual_workers.py](https://github.com/Motoki0705/tennis-lab/blob/11c1396cebe06885c9489626990be82a592c908e/.agents/ops/parent/manual_workers.py#L541) [worker_monitor.py](https://github.com/Motoki0705/tennis-lab/blob/11c1396cebe06885c9489626990be82a592c908e/.agents/ops/parent/worker_monitor.py#L239)

### 3. キューの正規の再待機と終端記録の欠落を正しく扱えない
静的な推論。`queue_result` は `jobs/`、`running/` の順に調べます。キュー停止時に起動直前のジョブが `running/` から `jobs/` に戻るため、走査の途中で移動すると、登録済みのIDを未知と誤判定して監視を打ち切ります。また、終端ディレクトリへ移動した直後に完了記録の書き込みが失敗すると、キュー監視に期限がなく、異常を示さず待ち続けます。移動中と欠落を区別し、終端記録が確定しない場合は明示エラーを返してください。ファイル読取境界で両方の順序を再現できます。 [worker_wait_sources.py](https://github.com/Motoki0705/tennis-lab/blob/11c1396cebe06885c9489626990be82a592c908e/.agents/ops/parent/worker_wait_sources.py#L77) [training_queue.sh](https://github.com/Motoki0705/tennis-lab/blob/11c1396cebe06885c9489626990be82a592c908e/.agents/skills/training-queue/scripts/training_queue.sh#L1009)

### 4. 監視が正常終了しても `wait-worker` が異常終了し得る
静的な推論。`wait-worker` が `waiting` のスナップショットを読んだ直後に、監視が `done` または `needs_parent` を保存して終了すると、直後のPID確認は不在を返します。CLIは保存状態を読み直さず、成功結果や上限到達の理由を返す代わりに `followup_controller_unconfirmed` で終了します。PID不在を返す前に同じ後続処理の最新状態を照合し、終了記録を優先してください。待機側の読取と監視終了を同期させるケースで検出できます。 [worker_wait.py](https://github.com/Motoki0705/tennis-lab/blob/11c1396cebe06885c9489626990be82a592c908e/.agents/ops/parent/worker_wait.py#L222)

### 5. 長時間のキュー監視がmainの定期更新を止める
静的な推論。通常手順どおりメインcheckoutから手動委任すると、監視サービスも同じcheckoutの `guard_interactive_cli` を通り、`ops-code.lock` の共有ロックをプロセス終了まで保持します。上限時間のないキュー待機中は、実装ワーカーと親の会話が終了しても、`git-sync` が `checkout_jobs_running` で更新を見送り続けます。監視コードの版を固定しつつ、待機時間全体ではメインcheckoutの保守ロックを保持しない起動方法が必要です。 [worker_monitor.py](https://github.com/Motoki0705/tennis-lab/blob/11c1396cebe06885c9489626990be82a592c908e/.agents/ops/parent/worker_monitor.py#L322) [checkoutの排他契約](https://github.com/Motoki0705/tennis-lab/blob/523c4a9ad42c2d130d335ea2a67fdb9e0144dfc6/.agents/ops/lib/README.md#メインcheckoutと運用jobの排他)

### 6. mainの共有メモリに手動再開とローカル待機の例外が残る
静的な推論。指定mainに先行マージされた #2310 の `workers-do-not-wait-on-queue-jobs.md` は、親が完了イベントを受けて手動で `run-worker --resume` するよう指示し、子による起動直後の安定確認も許可しています。PRは別のgotchaだけを削除するため、手動委任をCLIが自動再開し、ローカルqueue投入後は子が終了するという新しい手順と両方がマージ後に残ります。親が読むSUMMARY・索引・詳細を `parent-session` の手順と整合させ、Colabの既存手順を維持してください。 [mainの共有メモリ](https://github.com/Motoki0705/tennis-lab/blob/523c4a9ad42c2d130d335ea2a67fdb9e0144dfc6/.agents/memory/decision/workers-do-not-wait-on-queue-jobs.md#L9) [新しい手順](https://github.com/Motoki0705/tennis-lab/blob/11c1396cebe06885c9489626990be82a592c908e/.agents/skills/parent-session/references/workers.md#ciローカルqueueの待機)

## 親からの補足
- 6: 現在の origin/main に #2310 が入っているので、まず branch を origin/main へ rebase する（競合は解消し、#2310 の意図＝Colab の既存手順は維持）。`.agents/memory/decision/workers-do-not-wait-on-queue-jobs.md` と SUMMARY・索引を、新しい parent-session の手順と矛盾しないよう memory skill に従って整理する。手順の正本は parent-session に置き、memory に同じ手順を二重に書かない。
- 2: 枠を保持し続けるか、再開イベントを保持して枠の再取得を待つかは、静かな打切りが起きない方を選び、選んだ理由をPR本文に書く。再開できない場合は親へ明示的に返す（needs_parent 相当の記録と wait-worker の結果）。
- 3・4: 異常は黙って待ち続けたり別結果にしたりせず、明示エラーまたは保存済み終了記録を優先して返す。
- 5: 監視コードの版固定と、メインcheckoutの保守ロックを待機全体で保持しないことを両立する。
- 1: 事前検査と実際のロック範囲を揃え、起動後の競合も親が観測できる（通知が出る）ようにする。

## このターンの終わり方
- 全件の修正・回帰テスト・関連 e2e/unit・pre-commit をローカルで通し、commit・push して終了する。**CIの完了を待たない。** sleep・照会ループ・gh run watch・gh pr checks --watch を使わない。
- 最終報告に、指摘ごとの修正内容・回帰テスト名・結果、新しい完全HEAD、未確認事項を書く。非公開パス・セッションIDを書かない。
- 作業時間の上限は5時間。修正方針に親・人間の判断が必要な食い違いがあれば、その指摘の理由と選択肢を返して終了してよい（他の指摘は進める）。
