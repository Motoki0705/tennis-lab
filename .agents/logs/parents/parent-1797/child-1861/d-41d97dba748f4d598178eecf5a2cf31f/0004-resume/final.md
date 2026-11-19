[PR #1872](https://github.com/Motoki0705/tennis-lab/pull/1872)を更新しました。branch `codex/review-reuse-adoption`、commit `f09c906cb`。

- **pr-1872-1**: 公開状態と終了確認を分離。未確認runを阻害し、公開済みrunには正式結果を変えない`exit_confirmation`を追記します。
- **pr-1872-2**: 初回保存前の窓を実wrapperで再現。runtime欠落時は、review IDに束縛したsupervisor証拠で終結できます。
- 追加5 node・11ケース。関連99ケース成功、故障注入2件を検出・復元。pre-commit・型検査も成功しました。
- [新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42855208270/attempts/1)はLint・CPU全10レーン・Pythonが初回成功。再実行なしです。

親による2指摘のresolve・merge判断・merge後main確認が残ります。