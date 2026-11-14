## 親の具体的な指示（#1933）

- `.agents/ops/worker_logs/schema.py` の `identity` で、repo表記の正規化を一か所にまとめ、`base.repo`・`head.repo`・`source.repo`・`html_url` のすべてに使う。
- 回帰テスト: 小文字のrepoと正規表記の `html_url` の組を受理する（修正前に失敗すること）。番号違い・repo違い・ホスト違い・スキームやパスの改変の `html_url` は `pr_identity_changed` で拒否する。
- 同じ種類の大小文字の扱いが、`publication.py` など同じ公開経路の他の比較にもあれば、範囲内で揃える。見つけたが範囲外と判断したものは報告する。
- 作業時間の上限は2時間。止めてよい条件: 受入条件を満たしPRのCIが成功した。

## この一括作業での共通の制約

- 親 #1968 は、元々 `issue:standalone` だった12件を同時に解消している。他に11のworkerが並行して動いている。
- pytestは変更に関係するnodeに絞り、`.agents/ops/pytest_locked.sh <対象> -q`（-n 4固定）で実行する。repo全体の一括実行はせず、全体はCIで確かめる。
- GPUを使う処理は training-queue skill 経由でだけ実行する。
- PRは担当issueを閉じる場合に `Closes #<担当issue>` を付け、本文は日本語で書く。PRのmerge、issueのclose、親issueへの投稿、systemctl・timerの操作は行わない。issueやPRへの公開の報告文が必要なら、最終メッセージに案文を書く。
- 最終メッセージは次の見出しで書く: `## 結果`（受入条件ごとの状況と証拠）、`## branch・commit・PR・CI`、`## issueへの報告案`（不要なら「なし」）、`## 親への判断依頼`（なければ「なし」）。
