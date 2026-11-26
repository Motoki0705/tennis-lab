`origin/main` をマージし、knowledgeの連番競合は外部ボールrunを `000098` に採番して解消しました。

FP32の失敗は、旧テストが履歴fixtureに現行の `inventory.py` のハッシュを混ぜ、30ファイルの完全一致条件から外れたためです。`main` の履歴fixtureを取り込み、当時のハッシュをGit原本と照合しました。完全一致時にキャッシュを再利用し、対象ファイルに変更があれば再利用を拒否することを確認しました。`digest-mismatch: error` はダウンロード設定で、集約ジョブの失敗理由はCPUテスト2件でした。

関連テスト25件、pre-commit、knowledge検査が通りました。実データとベンチマークは変更していません。

[PR #2462](https://github.com/Motoki0705/tennis-lab/pull/2462) を更新しました。新HEADは `669cd21156937bf899abcd96f796f8852427998d` です。CI待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2462,"head":"669cd21156937bf899abcd96f796f8852427998d","resume_on_success":false}}