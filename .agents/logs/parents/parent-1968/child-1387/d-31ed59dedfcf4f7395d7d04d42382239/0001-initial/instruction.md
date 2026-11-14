## 親の具体的な指示（#1387）

- インストール済みのCodex CLIの版で、`model_context_window` がプロジェクトの `.codex/config.toml` から読まれるか（信頼されたprojectの条件、worktreeでの適用）を、公式の設定リファレンスと実機で確かめる。
- 使うモデル（`agents.conf` の gpt-6.1-sol と gpt-6-astra）の対応上限を確かめ、512,000がそれを超えないかを確認する。超える場合や効かない場合は、設定を入れずに理由を報告する。
- 共通ランナー経由の起動（`agent_exec.sh`）で、この値が実効値として読まれるかを確かめる。`model_auto_compact_token_limit` との関係も確認する。
- 作業時間の上限は2時間。止めてよい条件: 受入条件を満たしPRのCIが成功した。または、設定が効かない・上限を超えると確かめた（証拠付き）。

## この一括作業での共通の制約

- 親 #1968 は、元々 `issue:standalone` だった12件を同時に解消している。他に11のworkerが並行して動いている。
- pytestは変更に関係するnodeに絞り、`.agents/ops/pytest_locked.sh <対象> -q`（-n 4固定）で実行する。repo全体の一括実行はせず、全体はCIで確かめる。
- GPUを使う処理は training-queue skill 経由でだけ実行する。
- PRは担当issueを閉じる場合に `Closes #<担当issue>` を付け、本文は日本語で書く。PRのmerge、issueのclose、親issueへの投稿、systemctl・timerの操作は行わない。issueやPRへの公開の報告文が必要なら、最終メッセージに案文を書く。
- 最終メッセージは次の見出しで書く: `## 結果`（受入条件ごとの状況と証拠）、`## branch・commit・PR・CI`、`## issueへの報告案`（不要なら「なし」）、`## 親への判断依頼`（なければ「なし」）。

## 環境の注意（2026-11-14 追記）

- 2026-11-13 18:20 JST にD: の容量不足でWSLが停止し、復旧した。repo の `data/`・`outputs/`・`ckpt/` と各worktreeの `outputs/` は E: 上のext4ディスク（`[local path]`）へ移され、元のパスはシンボリックリンクになっている。既存のパスはそのまま使える。D: 側に大きなファイルを書かない。
