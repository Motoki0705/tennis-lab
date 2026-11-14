## 親の具体的な指示（#1478）

- まず、現行mainのレビュー起動（`.agents/ops/review/`、共通ランナー `agent_runtime.py`・`agent_settings.py`）と、インストール済みのCodex CLI（版を記録）で、Web検索がどう扱われるか（既定の有効/無効、設定キー、read-only sandboxでの挙動）を確かめる。
- レビュー起動でWeb検索の扱いを明示する。無効にするか、有効にしても終了しない構成にするかを、確かめた事実から選び、理由をPRに書く。
- CLIがシグナルで終了した場合を、`invalid_output` と区別した理由コードで記録する。終了シグナルの番号も記録する。既存の理由コードの利用者（親CLI・レビューCLI・ops-report）への影響を確認する。
- 実CLIでの短い確認runを行う場合は、read-onlyで、最小のpromptにする。
- 作業時間の上限は4時間。止めてよい条件: 受入条件を満たしPRのCIが成功した。

## この一括作業での共通の制約

- 親 #1968 は、元々 `issue:standalone` だった12件を同時に解消している。他に11のworkerが並行して動いている。
- pytestは変更に関係するnodeに絞り、`.agents/ops/pytest_locked.sh <対象> -q`（-n 4固定）で実行する。repo全体の一括実行はせず、全体はCIで確かめる。
- GPUを使う処理は training-queue skill 経由でだけ実行する。
- PRは担当issueを閉じる場合に `Closes #<担当issue>` を付け、本文は日本語で書く。PRのmerge、issueのclose、親issueへの投稿、systemctl・timerの操作は行わない。issueやPRへの公開の報告文が必要なら、最終メッセージに案文を書く。
- 最終メッセージは次の見出しで書く: `## 結果`（受入条件ごとの状況と証拠）、`## branch・commit・PR・CI`、`## issueへの報告案`（不要なら「なし」）、`## 親への判断依頼`（なければ「なし」）。

## 環境の注意（2026-11-14 追記）

- 2026-11-13 18:20 JST にD: の容量不足でWSLが停止し、復旧した。repo の `data/`・`outputs/`・`ckpt/` と各worktreeの `outputs/` は E: 上のext4ディスク（`[local path]`）へ移され、元のパスはシンボリックリンクになっている。既存のパスはそのまま使える。D: 側に大きなファイルを書かない。
