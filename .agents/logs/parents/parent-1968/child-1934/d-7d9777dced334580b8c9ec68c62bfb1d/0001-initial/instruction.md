## 親の具体的な指示（#1934）

- まず、担当issueの再現手順（timeout後の保存物の破損 → hold保存 → 修復 → tick・RetryWorker）を再現する回帰テストを書き、修正前に失敗することを確かめる。
- 修復後に完成receiptを確定する正式な経路を一つに決める（tickでの再検証か、recoverの明示操作か）。選んだ理由と、もう一方を採らなかった理由をPRに書く。
- 元のhold判断（理由・attempt）を完成確定後も読めること、未修復のままでは完成させないことをテストで確かめる。
- 注釈の自動継続の利用手順に影響する場合は、annotation-workflow skill と対象READMEの記述を更新する。
- 並行作業: #1792 も注釈の自動継続（`src/tasks/annotation`、`src/tasks/base/annotation/workers/`）を調べている。共通の箇所を変える必要があれば、差分を最小にする。
- 作業時間の上限は4時間。止めてよい条件: 受入条件を満たしPRのCIが成功した。または、人間の判断が要る事項が残り、それ以外の独立した作業を終えた。

## この一括作業での共通の制約

- 親 #1968 は、元々 `issue:standalone` だった12件を同時に解消している。他に11のworkerが並行して動いている。
- pytestは変更に関係するnodeに絞り、`.agents/ops/pytest_locked.sh <対象> -q`（-n 4固定）で実行する。repo全体の一括実行はせず、全体はCIで確かめる。
- GPUを使う処理は training-queue skill 経由でだけ実行する。
- PRは担当issueを閉じる場合に `Closes #<担当issue>` を付け、本文は日本語で書く。PRのmerge、issueのclose、親issueへの投稿、systemctl・timerの操作は行わない。issueやPRへの公開の報告文が必要なら、最終メッセージに案文を書く。
- 最終メッセージは次の見出しで書く: `## 結果`（受入条件ごとの状況と証拠）、`## branch・commit・PR・CI`、`## issueへの報告案`（不要なら「なし」）、`## 親への判断依頼`（なければ「なし」）。
