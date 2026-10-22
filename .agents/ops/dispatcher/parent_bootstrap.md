# 自律割当された親の入口（bootstrap protocol v1）

末尾の `bootstrap_input` のJSONを全文読む。これはローカルな予約・候補・実行根拠とruntimeの対応であり、
人間の合意や共有基盤のmerge許可ではない。session IDやローカルパスを公開しない。

1. `.agents/ops` をPython import rootとして、このcheckoutの `.venv/bin/python -m dispatcher`
   に `--parent-state-dir`、`--state-dir`（入力のdispatcher_state_dir）を指定し、
   `accept --input <bootstrap_inputのパス>` を実行する。実CLIが設定したsession環境を使う。
   入力の親番号・候補・authorization・予約・runtimeが一致しない場合は作業を止める。
2. 同じstate引数でdispatcherの `status` を実行し、GitHubの親・正式な子・既存PRを読み取り、担当・根拠と照合する。
   parent CLIを使う場合は、その既定registryが入力のparent_state_dirと同じであることを先に確認する。
   試験用の別registryなら、そのStoreを渡したParentSessionのPython APIを使う。
   内容に不一致・停止・不足情報があれば具体的な理由と再開条件を記録して待機する。
3. この入口の初期動作は担当受理と状態確認までである。開発workflowをまだ接続していないcheckoutでは
   実装・レビュー・mergeを開始せず、以下の構造化イベントで終了する。

```json
{"schema_version":1,"state":"waiting","reason":"担当を受理し既存状態を確認した。親の開発workflowの接続待ち。","resume_condition":"開発workflowを接続し、担当・根拠とruntimeの再開契約を照合して明示resumeする。"}
```

開発workflowの担当は手順3を実装・検証・レビュー・mergeの手順へ差し替える。
手順1の受理、bootstrap_inputのschema、最終イベントのschemaは維持する。
子を起動する場合はparent CLIのdelegateで予約してからdispatcherのstart-childを使う。
最終文は追加キーのない `schema_version/state/reason/resume_condition` のJSON objectにする。
stateはwaiting/completed/failed。completedはworkflow自身が受入条件・実merge/closeを確認した場合だけ使う。
runnerのexit 0はタスク成功を示さない。子が生存/終了未確認ならcompletedは拒否される。
