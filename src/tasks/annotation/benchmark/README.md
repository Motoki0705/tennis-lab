# 固定予測からの並列レビュー実行

`BenchmarkRunner` は保存済みball予測を、10→25→50の条件別review台帳へ参照登録して
`BallReviewWorker` を動かす。現在の対応範囲はballレビューのみ。画像・編集は
通常controllerと共通の `ServiceBroker` / `Operations` を通じ、CPUの `render` 枠で実行する。
専用 `benchmark-stage` はreview serviceだけを受け付け、planから条件別layoutを決定する。
worker側のCLIは自身のattempt inboxへ要求を書き、注釈の保存先は親が検証する。

入力・条件・測定の契約は[共通benchmark](../../base/annotation/benchmarks/README.md)が正本。
`BenchmarkRuntime` は比較仕様参照、全条件共通の `RuntimeSettings`、観測間隔を固定する。
review/renderそれぞれのPSS・保存予約・timeout、Codex binary/home、service回数上限を明示する。
既存RuntimeSettingsのqueue参照も検証するが、このrunnerはGPU/queueへ投入しない。
planにはCodexの `review` とCPUの `render` 枠が必要で、資源scopeは既存のものを再利用する。

~~~bash
.venv/bin/python -m src.tasks.annotation schema --type benchmark_runtime
.venv/bin/python -m src.tasks.annotation benchmark-status --runtime /absolute/runtime.json --sha256 HASH --condition workers-10
.venv/bin/python -m src.tasks.annotation benchmark-run --runtime /absolute/runtime.json --sha256 HASH --condition workers-10 --max-seconds 600
~~~

conditionにはComparisonSpecが参照するConditionSpecの `condition_id` を指定する。
全条件共通設定は `campaigns/<benchmark>/benchmark/runtime.json` へ不変保存し、途中での
設定差替えを拒否する。先行条件が全workerの採用と完了観測を持つまで、次条件を開始しない。
`workers_complete` は当該条件のレビュー完了であり、設定人数の同時稼働・品質合格を意味しない。
比較の受入状態は引き続き `unverified` とする。

`benchmark-tick` は1回の受付・照合・service処理を行う。`benchmark-run` はそれを反復する。
同じruntime/conditionで再開し、完了したworkerを再起動しない。runの時間上限は親の処理時間で、
受付済みsupervisorは自身のtimeoutと資源監視に従う。親の実行を継続してworkerのservice要求を
処理する。予算停止や観測件数上限では新規受付を止め、稼働workerの停止を要求する。
supervisorが起動途中などでまだ停止確認できないworkerは `stopping` と対象IDを返し、
次のtickで確認を続ける。停止要求だけで全プロセスが終了したとは報告しない。

保留・失敗を自動再試行しない。通常CLIと同じRecoveryInputを固定rootへ保存し、
`benchmark-recover --runtime PATH --sha256 HASH --condition ID --operation PATH` を使う。
許可するのは当該cohortの `retry_worker` と `resume_workers` のみ。再試行元attemptを照合し、
理由と結果を不変保存する。worker受付の再開は共有予算のstopを解除しない。

条件の完了は `campaigns/<campaign>/benchmark/complete.json` に、入力条件・共通runtime・
全workerの採用結果・最後の完了観測を結んで保存する。公開・学習の完了receiptには変換しない。
追跡/pose・公開・学習の工程別測定、playerレビュー、task品質の評価は後続の実装範囲。
CPU合成fixtureの通過は実動画・実Codexの10/25/50並列試験を代替しない。
