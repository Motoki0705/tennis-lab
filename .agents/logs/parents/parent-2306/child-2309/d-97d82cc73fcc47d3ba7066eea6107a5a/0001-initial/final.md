[#2309に調査結果を記録しました](https://github.com/Motoki0705/tennis-lab/issues/2309#issuecomment-7029457686)。**呼び鈴は `PreInvocation` での未読提示を推奨します。** ローカルRPCで直接投入できましたが、`[local path]` 後にも旧会話でターンが始まることを実測しました。

`agy` 1.5.13 と Herdr 0.9.5 を、一時HOMEと外部通信を遮断した環境で検証しました。モデル応答はループバック上の固定応答を使い、課金と既存セッションへの送信は発生していません。

### 1. 手動親の識別

| 実測操作 | 観測結果 |
|---|---|
| PTYで `agy` を起動し、`run_command` を実行 | フックの `conversationId` と環境の `ANTIGRAVITY_CONVERSATION_ID` が一致。コマンドには別UUIDの `ANTIGRAVITY_TRAJECTORY_ID` も渡った。PTYの `!` コマンドには会話IDが渡らなかった。 |
| `agy -p … --output-format stream-json`、`--continue`、`--conversation A` | 新規は別UUID、再開は同じUUID。printにも同形式の識別子が入る。対話から別の `agy -p` を起動すると、祖先に2つの `agy` が現れた。 |
| 対話内の `invoke_subagent` と `run_workflow` | 子は別UUIDだが、祖先の `agy` は親と同じPID。そのPIDが保持する `presence/*.lock` は親UUIDのものだけだった。 |
| 子UUIDを `agy --conversation` で開き直す | 子自身のロックを取得した。再開後にターンを実行しても、`GetConversationMetadata` の `rootConversationId` は元の親を指した。ワークフローの子はSQLite上の親IDが空で、深さも `0` だった。 |
| `agy acp --stdio` と `--process-per-session` | どちらも会話・trajectory UUIDを取得。分離モードの祖先は `agy acp --worker ← agy acp --process-per-session`。 |
| `agy --sandbox` | `run_command` に識別環境は届いたが、ツールから見える祖先は `exe`（PID 1）まで。フック側では前景 `agy` を観測できた。 |

`agy:<UUID>` は、環境の会話ID、単一の前景TUI祖先、そのPIDが実際に保持する会話ロックを照合して決める案です。親CLIの入口では、読み取りRPCの `rootConversationId` と `trajectoryId` も照合すれば、子履歴の開き直しを拒否できます。`[local path]` はスキル展開まで確認し、有料プランのworker・verifierは未計測です。 [子実行の実測]([local path]) · [ACP公式仕様](https://antigravity.google/docs/cli/acp) · [Swarm公式仕様](https://antigravity.google/docs/swarm)

### 2. フックと会話切替

プロジェクトの `.agents/hooks.json`、ユーザーの `[local path]`、pluginの `hooks.json` で発火しました。フックの実行cwdは設定ファイルの親ディレクトリで、JSONの `cwd` はworkspaceでした。JSONには `hookEventName` と `conversationId` があり、新規・再開・クリアを示す `source` は観測されませんでした。

PTYの `[local path]` では次の入力前に `statusLine.conversation_id` とロックが A→B に変わり、`[local path]` で B→A に戻りました。`SessionStart` は新規会話の初回ターンで発火し、再開では `PreInvocation` が発火しました。`Stop` はターン終了時に発火し、`[local path]` 前後で件数は同じでした。設定した `SessionEnd` の発火もありませんでした。表示連携には、起動単位の `agy --settings <JSON>` で `statusLine` を設定する案です。永続設定への追加は既存の読み取り専用ランナーの設定検査に抵触します。 [会話切替の実測]([local path]) · [agy_headless.py](https://github.com/Motoki0705/tennis-lab/blob/b9219c2a285b8472c5e9abf62d191928b11de8bc/.agents/ops/lib/agy_headless.py#L131) · [Google Hooks](https://antigravity.google/docs/hooks?tab=cli)

### 3. 呼び鈴

`SendUserCascadeMessage(WHEN_IDLE)` は、認証なしで `401`、正常な対話宛てで `200 {}` を返しました。実行中に送ると先行ターンの終了後に処理されました。一方、**前景がBに切り替わった後も旧Aへの送信は `200 {}` で、Aの `PreInvocation` が発火しました。** `herdr agent prompt` では、人が入力途中の下書きと送信文が一つのプロンプトになりました。 [RPC実測]([local path]) · [Herdr実測]([local path])

TUIのQueued Messagesは動作しましたが、外部書込用キューは確認できず、`agy help queue` は `unknown subcommand` でした。ACPで既存UUIDを読み込んだ場合も実行経路は別プロセスです。Remote Controlには同一アカウントのブラウザー経路があり、掲示板用のローカル公開APIは確認できませんでした。未読提示案では、検証済みルートの次の `PreInvocation(invocationNum=0)` に件数を渡し、本文取得と既読は `board` の操作で扱います。直接投入の採否は、旧会話を起動できる実測を踏まえて親に返します。 [Queued Messages](https://antigravity.google/docs/settings?tab=cli) · [Remote Control](https://antigravity.google/docs/remote-control/)

### 4. Herdrと変更箇所

`herdr pane list` は実機で3件を `agent="agy", agent_status="idle"` と報告しました。一時Herdrでは、標準integrationが `PreInvocation` で登録した会話参照と管理ラベルが `[local path]` 後もAのまま残り、`[local path]` 後に前景がbashへ戻ってもラベルが残りました。同じsourceへの `pane report-metadata` で終了後に消去できました。`allow_move` は、新paneが対象会話のロックを保持し、旧paneの空きを確認できたTUIイベントに限って許可する案です。 [終了後の消去実測]([local path]) · [Herdrの検出定義](https://github.com/herdrdev/herdr/blob/2ecf19a92fc3a7fff75b6ce3655ef04f4625ec01/src/detect/manifests/antigravity.toml#L1)

- **識別と担当** — `lib/identity.py:10,38` にルート証明を追加し、`parent/cli_impl.py:489,877,1175` と `parent/session.py:257,425` の担当操作を再利用。
- **表示** — `herdr_labels/service.py:185,320`、`coordination.py:85,223`、`client.py:71` に会話切替と終了の同期を接続。終了は実プロセスの終了イベントで扱う。
- **掲示板** — `board/service.py:215,334`、`messages.py:45`、`delivery.py:248`、`receivers.py:207`、`display.py:326` を接続。`storage.py:76,177` に既存DBを保持する移行を足し、未読提示待ちを配送失敗と区別する。
- **追跡規約** — `.github/labels.yml:38` と `review/review_policy.py:18` に `agent:agy`・`agy/` を追加。`dispatcher/README.md:366` と関連skillに手動親の範囲を記載。

### 5. 実装時の検証

`tests/e2e/agents_ops/` の既存fixtureを使い、同一PIDの子会話、別TUIで開き直した子、print・ACP・ネスト、重複pane、切替や終了後に届く古い処理を検証する計画です。親の登録・引継ぎ・解除、掲示板の投稿・未読・既読、現行DBからの移行、著者ラベルの矛盾検出を通します。terminal sandbox内からも親操作を受け付ける場合は、ホスト側の祖先を確認できる経路が必要です。

調査時のHEADは `b9219c2a2`、branchは `codex/parent-2306-issue-2309-primary` です。Herdr初回プローブでworktree内にできた一時出力は退避済みで、`git status` はcleanです。実装・コミット・push・PR作成は行っていません。