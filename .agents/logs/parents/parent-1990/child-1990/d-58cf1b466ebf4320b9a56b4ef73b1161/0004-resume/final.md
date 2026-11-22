[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023) をdraftのまま更新しました。`codex/storage-tiering` / `b1afe312d`。対象HEADの[Python CIは成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43620065988/job/132528811796)しています。

- CapPrm非包含・同一namespace・実際の権限拒否を確認する除外を追加。件数、comm、根拠の種類、両方の能力値を記録し、READMEを3言語で更新しました。
- **9ケース追加、関連553件成功[local path]。故障注入、復元後の検証、pre-commitも通過しました。
- 実機の小単位index dry-runは成功し、atimeを含むmetadataは不変でした。

**実機受入は未達です。** 「同じuser namespace内」を実行時の一致確認条件として実装しましたが、`[local path]`にも同じptrace検査がかかり、EACCESになります。CapPrm非包含は確認できてもnamespace一致を確認できず、systemd 1件の失敗が残りました。[kernel実装](https://github.com/torvalds/linux/blob/v6.6/fs/proc/namespaces.c)

親判断をお願いします。同一namespaceを**運用前提**とし、未確認を記録してCapPrm非包含＋実拒否で除外する案を推奨します。一致確認を必須にする場合は、現在の失敗を維持します。

実機の配置切替・実削除・unit操作は未実施です。validatorは指定なし、試行／完了0/0です。