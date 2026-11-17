[PR #2206](https://github.com/Motoki0705/tennis-lab/pull/2206) を作成しました。

- 旧前提のmemory 2件と索引・SUMMARYを削除し、CI必須判断のリンクを修正。
- branch: `codex/memory-old-review`
- commit: `a59aba6b164d351b626929f96c158cd6f9fb856e`
- テスト150件成功（18.73秒）、実機probe 5項目成功。pre-commit正常終了（ruff/mypy対象なし）。
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43899563998) はqueued。完了を待たず終了します。
- validatorは指定なし、試行・完了とも0回。

実CIログ取得はghの版不足で未確認です。レビューからmerge後確認までの実GitHub検証は、親の`review-request`へ引き継ぎます。