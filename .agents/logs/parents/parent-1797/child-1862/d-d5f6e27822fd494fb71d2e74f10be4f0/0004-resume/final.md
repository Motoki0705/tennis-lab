同じ [PR #1888](https://github.com/Motoki0705/tennis-lab/pull/1888) で2件を修正しました。未mergeです。

- **pr-1888-1**: reviewedを含むfailed結果にも実効値・runtime digest検証を必須化。
- **pr-1888-2**: 検証済み候補をhandoff・recoverで保持し、親の明示確認までgateを停止。
- 追加15ケース（unit 7、E2E 8）、既存対象テスト、pre-commitが成功。故障注入3件を検出し、復元後も成功。
- branch: `codex/review-model-selection`、commit: `252562975`
- [新HEADのCI #3976](https://github.com/Motoki0705/tennis-lab/actions/runs/42949323215) は13ジョブすべて成功。再実行0回。

実モデル比較は #1863 に残しています。正式な指摘解消とmergeは親へ返します。merge後mainのCIは未確認です。