[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023) を更新しました。branch `codex/storage-tiering`、HEAD `3e5fdbeaa`。worktreeはclean、`Closes #1990`を維持しています。

- **pr-2023-1：再現しました。** 別namespaceの重ねmountで、修正前はevict・旧ミラー退役が成功扱いになりました。単位内mountのinode不一致をunknownとして拒否し、祖先・単位外の除外は維持しました。
- **pr-2023-2：再現しました。** hydrateに加え、mirror・verify・index・旧manifest移行でも保存済みSHA-256を照合します。不一致時はD:へ公開せず、manifestも更新しません。

関連 **661件成功（62.00秒）**、純増12ケース。故障注入で13件失敗、復元後14件成功。pre-commit成功。[対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43661484265/job/132656078786) はPythonを含む全14ジョブ成功です。実mount 10件はCIでは権限制約でskip、ローカルでは全件成功しました。

実機のckpt小単位と限定[local path] 0。atime・実コンテナのmetadata・symlinkは不変、状態directory作成なしです。CI用の祖先bindでも止まらないことを確認しました。

README 3言語・PR本文を更新済みです。配置切替・本番実削除・unit・sudo操作は未実施。追加validatorは指定なし、試行／完了0/0です。