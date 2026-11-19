[PR #2182](https://github.com/Motoki0705/tennis-lab/pull/2182)へ反映済みです。branch `codex/scene-inference-runtime`、commit `86cddbe39`。Ready・競合なしです。

- 指摘1・2・3・5を修正。全nodeのA/B/C確認表をPR本文へ記録しました。
- CPU **453件成功（追加20ケース）**、pre-commit、[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44296659152)、[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44296659009)も成功しました。
- 指摘4は親判断で対象外として制限を明記しました。cache identity変更により旧成果物はexecuteで再生成が必要です。
- 追加GPU実行は0回です。

統合レビュー・merge・closeを親へ返します。