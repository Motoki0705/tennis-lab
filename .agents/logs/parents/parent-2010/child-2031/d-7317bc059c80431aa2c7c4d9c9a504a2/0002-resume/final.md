`inspect-worker`に仕様衝突があり、指定された停止条件に該当するため親判断を返します。

mainは読み取り専用として書込み・lock作成を禁止しています。一方、#2031は照会前のintent保存を要求します。統合後の実CLIはStore同一性チェックで失敗し、clientを結び直すだけでは読み取り専用契約を破ります。

選択肢は次の2つです。

- **A（推奨）**: inspectだけ永続intent保存の例外とし、照会前後のsnapshotでjob/runを照合する。
- **B**: inspectの観測intent書込みを、読み取り専用契約の例外として認める。

ローカルrebase候補は `8dfe4b319b`、branchは `codex/dispatcher-triage-io`、作業ツリーはcleanです。関連165件は164成功・当該CLI 1失敗、探索・鮮度11件とruff/mypy・pre-commitは成功しました。

[証拠記録](https://github.com/Motoki0705/tennis-lab/issues/2031#issuecomment-6885859579)を残しました。[PR #2056](https://github.com/Motoki0705/tennis-lab/pull/2056)へのforce-push、新HEADのPython CI、本文更新は未実施です。validatorは指定なし、試行0・完了0回です。

A/Bの判断をお願いします。