## 親の具体的な指示（#1641）

- 導入ガイドに、公式Driveの上限超過の判別（gdownのメッセージ、直接URLの `Google Drive - Quota exceeded` のHTML）と、次の取得手順を加える。取得したファイルは `src/tennis_scene/installation_assets.py` の固定SHA-256・サイズで検証する経路を示す。
- 代替の取得先（例: Hugging Face の camenduru/GVHMR のLFS）は、現在もSHA-256・サイズが一致するかを、LFSのメタデータまたは実際の取得で確かめてから載せる。配布条件・ライセンスを確認し、不明なら「利用者が自分で判断する取得先」として扱いを明記する。
- 正しいファイルと異なるbytesのファイルで、preflightの合否が分かれるテストを確認または追加する。
- 共有memoryの判断により、公開向けの文書は英語版を既定とし、日本語版 `.ja.md` と簡体字中国語版 `.zh-CN.md` の3言語を維持する。文書にハッシュ値を書かない（固定値はコードを参照させる）。
- 作業時間の上限は3時間。止めてよい条件: 受入条件を満たしPRのCIが成功した。

## この一括作業での共通の制約

- 親 #1968 は、元々 `issue:standalone` だった12件を同時に解消している。他に11のworkerが並行して動いている。
- pytestは変更に関係するnodeに絞り、`.agents/ops/pytest_locked.sh <対象> -q`（-n 4固定）で実行する。repo全体の一括実行はせず、全体はCIで確かめる。
- GPUを使う処理は training-queue skill 経由でだけ実行する。
- PRは担当issueを閉じる場合に `Closes #<担当issue>` を付け、本文は日本語で書く。PRのmerge、issueのclose、親issueへの投稿、systemctl・timerの操作は行わない。issueやPRへの公開の報告文が必要なら、最終メッセージに案文を書く。
- 最終メッセージは次の見出しで書く: `## 結果`（受入条件ごとの状況と証拠）、`## branch・commit・PR・CI`、`## issueへの報告案`（不要なら「なし」）、`## 親への判断依頼`（なければ「なし」）。

## 環境の注意（2026-11-14 追記）

- 2026-11-13 18:20 JST にD: の容量不足でWSLが停止し、復旧した。repo の `data/`・`outputs/`・`ckpt/` と各worktreeの `outputs/` は E: 上のext4ディスク（`[local path]`）へ移され、元のパスはシンボリックリンクになっている。既存のパスはそのまま使える。D: 側に大きなファイルを書かない。
