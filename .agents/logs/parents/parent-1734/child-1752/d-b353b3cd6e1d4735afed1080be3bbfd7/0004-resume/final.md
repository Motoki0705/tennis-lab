[Draft PR #2227](https://github.com/Motoki0705/tennis-lab/pull/2227) を更新しました。branchは `codex/video-batch-cli`、HEADは `d392dd375`、baseは指定どおりです。worktreeはcleanです。

- 指定設定で3clip・114nodeが全段完走。各side 2 identity、C3管理7resourceは各1load/unload。実行31.9分、RSS peak 8731MiB、NVML peak 6198MiBでした。
- 単一027との最終scene 29配列・metadataは完全一致。ただしBall補助evidence 9配列が従来許容を超え、**全component一致は未達**です。許容値変更・GPU再試行はしておらず、採否を親レビューへ残しています。
- CPU 399件、ruff/mypy、最終HEADの[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44292726151)・[Knowledge](https://github.com/Motoki0705/tennis-lab/actions/runs/44292726272) は成功しました。
- 全69clipの条件付き見積りは25.27〜25.88時間。[本番用コマンド](https://github.com/Motoki0705/tennis-lab/blob/d392dd375/knowledge/runs/run-i1752-shared-fp16-20261118/full69-command.txt) は保存済み、**未投入**です。

依存PRの後続レビュー修正は未取り込みです。独立validatorは指定0・試行0・完了0。merge・closeは行っていません。