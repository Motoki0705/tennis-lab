[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023) の残作業を完了しました。branchは `codex/storage-tiering`、HEADは `f17898523`。worktreeはclean、`Closes #1990`を維持しています。

- ckptをトップレベルdirと直下ファイルの単位に分割しました。ckpt rootは交換せず、直下ファイルの旧inodeは履歴へ保持します。
- mount参照検査は削除経路に限定し、読取り時のfd/cwd/mmap・snapshot照合は維持しました。
- CI bind元を共通Configから導くE:へ固定し、README 3言語と移行手順を更新しました。

修正前の5件・権限差1件の失敗記録を確認済みです。関連 **649件成功（55.70秒）**、最終補正後6件も成功、pre-commit成功。[対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43653228596/job/132631321370) はPythonを含む全14ジョブ成功です。実mount 8件はCIでは権限制約でskip、ローカルでは全件成功。任意validatorは指定なし、試行／完了0/0です。

再起動後も、ckpt小単位（計4,348,832B）の通常CLI dry-runと削除用メタデータ検査はunknown 0でした。atimeを含むpayload・実コンテナのmetadata、symlinkは不変、状態directoryの作成なし。既存[local path] QAも成功しました。

第2段では旧manifestを`index ckpt`で保存し、子dirを個別にindex／hydrateします。現在のbindは既にE:なので、**切替前の再インストールは不要**です。更新は、D:切替後にassetsを次回起動するまでに適用します。

人間がmain反映後の保守時間中、CI/runner queueのjob終了を待って実行する手順例です。活動中queueの停止は学習を終了させるため、先に空にします。

```bash
runner_service="$(sudo cat [local path])"
sudo systemctl stop "$runner_service" tennis-lab-training-queue.service
sudo bash scripts/github_actions/install_self_hosted_runner.sh
sudo [local path] start
# installerが起動するため、配置切替中は再停止
sudo systemctl stop "$runner_service" tennis-lab-training-queue.service
# D:切替・hydrate・受入確認後
sudo systemctl start tennis-lab-training-queue.service "$runner_service"
```

配置切替・本番実削除・sudo・unit操作は未実施です。rootファイル集合の公開はファイル単位で、途中失敗時は履歴とjournalによる手動復旧が必要です。