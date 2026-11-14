[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023) をdraftのまま更新しました。`codex/storage-tiering` / `2b3ddeefa`。対象HEADの[Python CIは成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43622589073/job/132535956968)しています。

- 同一namespaceを運用前提に変更。uid_mapの不一致は拒否し、読取不能は非fatalとして記録。一致してもnamespace同一性は未確認と記録します。READMEも3言語で更新しました。
- 能力差テスト13件（純増4）、関連検証 **557件成功[local path]。故障注入・復元後検証・pre-commitも成功しました。
- 実機ではプロセス検査の不明 **0件**。systemdを能力差で除外でき、scan/cycleに残る失敗は配置未切替の3件だけです。小単位7ファイルのindex dry-runも成功し、atime等は不変でした。

判断待ちはありません。配置切替・実削除・unit導入は第2段に残ります。validatorは指定なし、試行／完了0/0です。