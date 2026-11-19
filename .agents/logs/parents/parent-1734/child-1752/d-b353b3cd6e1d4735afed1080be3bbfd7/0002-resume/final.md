[PR #2227](https://github.com/Motoki0705/tennis-lab/pull/2227)（Draft）を更新しました。  
branch `codex/video-batch-cli`[local path]`920945355`、baseは指定のC3、worktreeはcleanです。

- 一括CLI、共有コート準備、入力検証、clip単位公開、失敗分離、再開、進捗記録を実装しました。
- CPU **536件成功**、再現検査14件成功、pre-commit成功。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44135718372)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44135717725)も対象HEADで全成功です。
- **GPU受入は未完です。** clip_000が人物対応の曖昧さで停止し、完成比較は0件でした。停止prefixは1200.933秒です。C3のハング疑いも把握し、追加GPUは投入していません。
- 全69clipの入力・共有コート照合は成功し、[実行コマンド](https://github.com/Motoki0705/tennis-lab/blob/92094535541c3664e2ea5f6259f2061b0fe0c00d/knowledge/runs/run-i1752-association-stop-20261118/full69-command.txt)を保存しました。本番は未実行です。参考43.3時間は失敗prefixの単純換算で、全工程の見積りではありません。

C1 branchの追加取り込みと再現検査の変更はPRに明記しました。GPU再開・比較subset・人物対応の扱いを親判断へ返します。validatorは指定／試行／完了 **0／0／0**です。