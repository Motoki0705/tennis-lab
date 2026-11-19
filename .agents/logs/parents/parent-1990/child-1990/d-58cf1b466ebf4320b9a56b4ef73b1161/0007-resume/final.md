[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023)を更新しました。branch `codex/storage-tiering`、commit `53e5e2812ce5743615bfcf12dd4ec6287aaa60c2`。作業ツリーはcleanです。

- **pr-2023-1：4ケースで再現・修正済み。** claim後のFDでwrite leaseを必須化し、保持中に照合、unlink直前・直後にleaseを確認します。取得不能・breakは失敗とし、削除済みentryと事故のphaseをjournal・auditへ記録します。
- **pr-2023-2：交換後のwrite/addの2ケースで再現・修正済み。** 旧E:も照合済み計画・lease・journal・recoverを通します。`shutil.rmtree`を除去し、未照合stageは保持します。README 3言語も更新しました。

**17ケース追加、関連595件成功（48.53秒）**。故障注入・復元後検証、pre-commitも成功しました。[対象HEADのPythonは成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43636131778/job/132581272068)、最新CIは全14ジョブ成功です。初回は計測APIのHTTP 502で失敗し、同HEADで再実行しました。integration 110件・追加17件は両実行とも成功。実mount 7件はローカル成功、CIでは権限不足でskipです。

残る前提は、consumerが保持領域のランダムtokenを探索・操作しないことです。lease確認とunlinkは別syscallで、その窓のopenによる後続書込みは復旧できない場合があります。検出したbreakは成功扱いしません。外部hardlinkが残ればpayloadも残ります。

実機は読取専用dry-runのみで、metadata不変を確認しました。配置切替・実削除・unit操作は未実施です。PRは第1段の指示に合わせ`Refs #1990`とし、修正確認・merge・第2段を親へ引き継ぎます。