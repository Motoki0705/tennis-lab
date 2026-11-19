[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023)を更新しました。branch `codex/storage-tiering`、commit `d0b1ee27ce4bec730a1fb597924adafd816cbbef`。作業ツリーはcleanです。

- **pr-2023-1：再現・修正済み。** 旧HEADで遅延write/read/add/replaceの4件が失敗しました。全entryの照合計画と削除対象を原子的な退避で結び付け、削除直前にhash・metadataを確認します。部分削除の記録、未照合payloadの保持、復旧中のミラー禁止も追加しました。
- **pr-2023-2：再現・修正済み。** 一時環境の実bind mountで4件が失敗し、checkpoint実体の削除も再現しました。基準mountから単位rootまでのmount ID検査と、checkpointディレクトリのdevice/inodeによる拒否を追加しました。

**21ケース追加、関連578件成功（41.25秒）**。故障注入・復元後検証、pre-commitも成功しました。README 3言語とPR本文を更新しています。

対象HEADの[CIは全14ジョブ成功、Python成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43630684577/job/132561873714)です。実mount 7件はローカルで全件成功し、CIではuser namespace作成権限がなくskipでした。必須のmount ID境界5件を含むintegration 93件はCIで全件成功しています。

実機dry-runは小単位の照合が成功し、atime等は不変でした。scan/cycleに残る失敗は配置未切替の3件だけです。実機の配置切替・実削除・unit操作は行っていません。修正後の正式再レビューと第2段は親へ引き継ぎます。追加の判断待ちはありません。