## 修正依頼（PR #2153、正式レビューの保存出力から）

レビューrun（parent-1734-e8b39802977749e78e4a9cf58a430041）は出力形式の不備で公開されなかったが、次の指摘は妥当なので修正する。

1. [major] `knowledge/nodes/tennis_scene/000042-run-dino-batch-api-20261116.md` のsessionフィールドに、実際の会話session IDが残っている。同じIDは `run.json`・`queue-repro-original.sh`・`gpu.log` にもある。ノードの `repro.command`、`inputs.json`、`results.json`、変異テストログなどには、ホーム・マウント先・worktreeの絶対パスが含まれる。
   - 公開コピーからsession IDを除く。
   - パスはrepo相対か、明示した環境変数（例: `$REPO_ROOT`、`$DATA_ROOT`）に置き換える。
   - 再現に必要なcommit・入力・checkpoint・binaryのhashは保つ。
   - 元の証跡が必要なら、Git外の非公開の場所（outputs配下）に置き、knowledgeにはhashだけを残す。
   - PR内の全ファイルを検索し、残りがないことを確かめる（`[local path]`、`[local path]`、`.claude/worktrees`、session IDの形式）。
2. knowledgeの連番について: 同じ親の別PR（#2154・#2155）も `tennis_scene` の連番42を使っている。#2153を先にmergeする予定なので、このPRは42のままでよい。

修正後に対象HEADのCI（`Python` と Knowledge を含む）が成功したことを確かめ、新しいHEADと確認結果を報告する。
