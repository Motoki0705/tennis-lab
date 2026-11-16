# PR #2184 再レビューの指摘の修正（#2176）

再レビュー（run parent-2157-2563831ac16e4d97acd1bf7068a77945）で、前回の指摘1・3・4・5は解消と確認されました。前回の指摘2の残りとして major が1件あります。修正して同じPRへpushしてください。HEAD 4017f9a1c のCIは15件すべて成功しています。

## pr-2184-1: #2157 方針(g)・完了条件9：自動修正の実起動でも取得範囲を限定する
`.agents/ops/dispatcher/worker.py`:38

静的推論です。修正予約ではProgressionGitHubを使いますが、実際の起動はRuntime.spawn → worker.pyで通常のGitHubを再生成します。targeted_history=Falseのため、claim → _validate_start → authorization → session.history → EvidenceIssues.findから `issues?state=all&labels=parent-evidence` の全ページを取得します。GET量が無関係な親の証拠Issue数にも依存し、#2176が求める全体一覧経路の除去を満たしません。追加テストは317行でbackend.dispatcher().claimを直接呼び、このプロセス境界を通していません。実workerのclaim・launchにも番号限定取得を維持し、その入口で一覧GETがないことを検証してください。

## 親の方針
- 実worker起動（Runtime.spawn → worker.py → claim → _validate_start → authorization → session.history）の経路でも、証拠Issueは保存済みの番号で取得し、`issues?state=all&labels=parent-evidence` などのrepo全体一覧を取らないでください。保存番号がない・取得できない場合は一覧へ戻らず、理由付きで止めてください（静かなフォールバック禁止）。
- テストは、実際のプロセス境界（dispatcherのworker起動プロセス）を通し、そのclaim・launchの入口で一覧GETがゼロであること、無関係な証拠Issue数を増やしてもGET数が変わらないことを確認してください。修正を外すと失敗することも確かめてください。
- 同じ全体一覧の取得が自動進行の他の起動経路（手動の修正起動、通知・再開）にも残っていないか確認し、残っていれば同じ方針で直してください。

push後に、修正内容・テスト・新しいHEADを報告して終了してください。CIの完了は待たなくてよい。
