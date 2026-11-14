[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023) を更新しました。`codex/storage-tiering` / `2909eb537`。対象HEADの[Python CIは成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43617119520/job/132519837266)しています。

- 非dumpable除外に、root所有・fd/maps/cwdすべての権限拒否・PID/UID等の不変を要求。件数・comm・根拠を記録し、READMEを3言語で更新しました。
- 12ケース追加。関連検証 **544件成功[local path]、checkout協調テスト **3件成功[local path]。pre-commitも成功しました。
- 実機では非dumpable **4件**を除外。小単位のindex dry-runは成功し、atimeを含むmetadataは不変でした。

**実機受入は未達です。** systemd（PID 544）だけは通常所有者のまま読取不能で、非dumpableの根拠が立ちません。指示どおり失敗を残したため、配置エラー以外の失敗が1件残っています。

親判断として、systemdを検査対象に残して読取権限を調整する案を推奨します（具体的方法は未検証）。別案の追加除外には新たな合意が必要です。

配置切替・実削除・unit導入／有効化は未実施です。validatorは指定なし、試行／完了0/0です。