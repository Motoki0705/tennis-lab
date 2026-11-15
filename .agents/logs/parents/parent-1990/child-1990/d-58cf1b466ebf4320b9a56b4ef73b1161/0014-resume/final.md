同じbranch `codex/storage-tiering`、commit `0a297c434`で更新・pushしました。worktreeはclean、`Closes #1990`を維持しています。[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023)

完了receiptに残る旧stageを、`recover`で履歴へ回収する方式を採用しました。境界検査で見つかったコピー前のintent不足、retained identity未記録、JSON一時ファイルの誤読、旧ckpt indexの再試行、孤立stageの誤成功も修正しました。READMEは3言語で更新済みです。

| 操作 | 境界数 | 結果 | 対応テストの代表 |
|---|---:|---|---|
| mirror公開 | 94 | 全件成功 | `test_mirror_new_persistence_boundary` |
| 旧ミラー退役 | 84 | 全件成功 | `test_mirror_update_removal_persistence_boundary` |
| hydrate | 30 | 全件成功 | `test_hydrate_persistence_boundary` |
| evict・hardlink | 96 | 全件成功 | `test_evict_persistence_boundary` |
| stage破棄・部分コピー | 160 | 全件成功 | `test_discard_mirror_removal_persistence_boundary` |
| ckpt直下ファイル公開 | 153 | 全件成功 | `test_checkpoint_mirror_persistence_boundary` |
| 旧ckpt index | 9 | 全件成功 | `test_index_legacy_persistence_boundary` |
| recover各状態 | 145 | 全件成功 | `test_recover_complete_stage_persistence_boundary` |
| 復旧176経路の再中断 | 2,090 | 全件成功 | `test_checkpoint_mirror_recovery_boundary` |

全2,861境界の個別結果と具体的なnode名を保存しました。各行を同一HEADのCI成果物と照合済みです。[全境界表](https://github.com/Motoki0705/tennis-lab/blob/0a297c4346092c11b557e0116a3902d5a657760d/tmp/storage-tier-review6-boundary-results.md)・[CSV](https://github.com/Motoki0705/tennis-lab/blob/0a297c4346092c11b557e0116a3902d5a657760d/tmp/storage-tier-review6-boundary-results.csv)

- 修正前HEADでは706境界中217境界で復旧契約が失敗しました。
- 対象HEADの関連**3,568件成功、905.31秒**。純増2,890件。pre-commit・ruff・mypyも成功しました。
- **Pythonを含む全14ジョブ成功**。実mount 10件はCIでは権限制約でskip、ローカルでは全件成功です。[CI結果](https://github.com/Motoki0705/tennis-lab/actions/runs/43680051376)
- 実機の読取専用QAはunknown 0、atime・metadata・symlink不変でした。

配置切替・実機recover/削除・unit・sudo操作は第2段に残ります。receiptのない旧stageには文書化した明示退避手順が必要です。validatorは未指定、実施0回です。