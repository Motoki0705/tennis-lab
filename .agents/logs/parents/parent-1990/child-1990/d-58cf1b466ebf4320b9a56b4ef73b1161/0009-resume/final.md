[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023) を更新しました。branchは `codex/storage-tiering`、HEADは `5aaf2e546`。`Closes #1990` を維持しています。

- 単位のfilesystem内pathをツール自身のmount情報から求め、祖先・外部のbindはstatせず除外数を監査します。単位内・判定不能は引き続き失敗にします。README 3言語も更新しました。
- 修正前の失敗を再現し、17ケース追加。関連 **626件成功（36.23秒）**、pre-commit成功。
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43647827408/job/132614791213) はPythonを含む全14ジョブ成功。実mount 8件はCIの権限制約でskip、ローカルでは全件成功です。
- 実機の既存小ペア（7ファイル・251,669B）でcycle／verify dry-run成功。D/Eの検査計8回、unknown 0。atime・metadata・symlinkは不変、状態ファイル作成なし。旧検査ではunknown 21件を再現しました。

実機検証は、既存ペアへprefix・inventoryを限定対応させた読取専用QAです。**通常CLIは配置未切替で停止するため、その受入確認は未完了です。**

また、ckpt全体はCI bindとrootが一致し、指定どおり検査対象になるためunknown 21件が残ります。第2段前に、CI bind・権限の調整またはckpt管理単位の判断が必要です。配置切替・実削除・unit操作は行っていません。