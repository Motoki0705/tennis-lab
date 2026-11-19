## 親の具体的な指示（#1943）

- 担当issueの提案どおり、手動親が `run-worker` で起動したrunの終了を、run記録（委任の `worker_runs`、共通ランナーの run dir・`result.json`、`agent_exec.sh status` が返すrunner/CLIの生存）で待つ入口を親CLIに追加する。プロセス名の文字列検索は使わない。
- `run-worker --detach` と `wait-worker` の組合せを採るかどうかは、実装時に判断し、理由をPRに書く。
- 受入条件の5項目すべてにテストを付ける。特に、待機コマンドのコマンド行に対象の子番号やrun-workerの引数が含まれても終了で返ること、強制終了したrunnerを「生存不明」として返すこと、待機中にロックを保持しないことを確かめる。
- parent README と parent-session skill の手動worker手順を、この入口を使う形に改める。会話の背景実行でこの入口を待つ手順を1つにまとめる。
- 並行作業: #1925（register の要約検査）と #1704（parent-session skill の分岐点の整理）も、同じ `.agents/ops/parent/` と `.agents/skills/parent-session/SKILL.md` を編集する。自分の差分は担当範囲の箇所に留め、無関係な節を書き換えない。
- 作業時間の上限は5時間。止めてよい条件: 受入条件を満たしPRのCIが成功した。または、人間の判断が要る事項が残り、それ以外の独立した作業を終えた。

## この一括作業での共通の制約

- 親 #1968 は、元々 `issue:standalone` だった12件を同時に解消している。他に11のworkerが並行して動いている。
- pytestは変更に関係するnodeに絞り、`.agents/ops/pytest_locked.sh <対象> -q`（-n 4固定）で実行する。repo全体の一括実行はせず、全体はCIで確かめる。
- GPUを使う処理は training-queue skill 経由でだけ実行する。
- PRは担当issueを閉じる場合に `Closes #<担当issue>` を付け、本文は日本語で書く。PRのmerge、issueのclose、親issueへの投稿、systemctl・timerの操作は行わない。issueやPRへの公開の報告文が必要なら、最終メッセージに案文を書く。
- 最終メッセージは次の見出しで書く: `## 結果`（受入条件ごとの状況と証拠）、`## branch・commit・PR・CI`、`## issueへの報告案`（不要なら「なし」）、`## 親への判断依頼`（なければ「なし」）。
