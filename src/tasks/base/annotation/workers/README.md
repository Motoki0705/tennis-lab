# ローカル注釈ワーカー

`WorkerPool` は task adapter を注入してローカル Codex を起動する。
ball/playerの意味・ラベルschema・レビュー方針は解釈しない。adapter が
`prepare` で指示を作り、`validate` で画像確認範囲・教師・hash等を検証して
成果物receiptを返す。共通層は `complete / deferred / not_applicable` と
採用receiptの有無を保存する。worker完了とdataset公開完了は別である。

`WorkRequest` はadapter版・工程・対象ID・入力artifactを固定する。
実行identityはadapter名と対象IDで決まり、campaign/モデルを変えても
同一対象を新規注釈として二重登録しない。adapterの具体実装は呼び出し側で
渡し、baseからtaskモジュールへのimportは設けない。

```python
resources = ResourceManager(layout, scope_ref)
pool = WorkerPool(
    layout,
    plan,
    (adapter,),
    resources=resources,
    worker_pss_bytes=worker_peak,
    worker_storage_bytes=worker_output_budget,
)
pool.register(request)
attempt = pool.start(request)  # 完了済みなら None、稼働/枠不足は WorkerBusy
state = pool.reconcile(request)
```

- 共有SQLiteと実行対象単位の `flock` を使用する。親が保持するlock descriptorを
  supervisorへ継承してから親のdescriptorを閉じる。親の終了・保存直前の中断で
  同じworkがもう一度起動することを防ぐ。supervisorが先に終了しても、残った
  子プロセスはPID/start ticks/session/process group/boot IDで追跡する。
- 別sessionのsupervisorがCodexとそのツールを所有し、終了・timeout・停止時に
  子孫を片付ける。保存PIDだけを根拠に `killpg` しない。pidfdで再利用PIDへの
  誤送信を防ぎ、ownerが不明瞭な場合は明示的に停止する。
- `gpt-6.1-sol` とplanのeffortを指定し、ユーザーconfig・rules・子agent・skills
  読込を無効にする。ツールのnetwork/GPUを無効にし、CPU内部thread数を工程から
  設定する。作業CWDは `attempt/work`、追加write領域もattempt内の宣言済みdirectory
  に限定する。入力、prompt、supervisorのreceiptはworkerのwrite領域に含めない。
  `/tmp` と継承したTMPDIRへの暗黙のwrite許可を除き、一時領域は `work/tmp` に置く。
  親指定のCodex state directoryをlaunchに固定する。productionでは起動前と
  supervisor側でcode commit/diffと未追跡ファイルを確認する。
  [公式の非対話実行仕様](https://learn.chatgpt.com/docs/non-interactive-mode) と
  インストール済みCLIのhelpを確認している。
- 成功はexit code 0・thread ID・`turn.completed`・失敗イベントなしをすべて要求する。
  その後adapterが機械検証し、immutableな結果コピーとreceiptを公開して台帳へ反映する。
  receiptと台帳の間で中断しても再検証・再起動せず同じreceiptを回収する。
  preview planからの教師採用は拒否する。
- 失敗・保留・timeout・cancelは自動retryしない。`start(..., retry=True)` を明示する。
  quotaイベントはcampaign全体の新規投入を止め、明示的な `resume_campaign` を要求する。
  admission時にも終了receiptを検証し、quota状態とcampaign停止を同じtransactionで保存する。
  回収はwork lock取得後に現在のattemptを読み、状態更新は検証したattemptと遷移元状態に限定する。
  完了と停止・retryが競合しても、確定済みreceiptや後続attemptを古い停止で取り消さない。
- Codex合計と同名工程の並列枠をcampaign横断で共有する。異なる上限が併存する場合は
  稼働中planと新規planの小さい上限を適用する。終了済みattemptは枠を返し、結果の
  validationはそのcampaignのadapterで再開する。productionでは
  [resources](../resources/README.md) のscopeとPSS/保存予約を必須とし、
  `CapacityUnavailable` と `BudgetStopped` を区別する。supervisorが資源leaseを継承し、
  稼働中は保存走査を待たない `watch()` を使い、終了時の回収を担当する。
  lease取得前に不変launchを保存する。資源受付後・worker台帳登録前に中断したattemptは、
  再開時に台帳へ失敗として回収し、明示retryでは新しいattempt/request IDを発行する。
  まだ資源を取得していないpending要求は同じIDで受付を再試行し、待機枠を重複消費しない。
  benchmarkでも資源要求は共有の資源台帳で照合し、workerの復旧状態は条件別台帳に記録する。

標準状態は `ready → running → complete`、未回収終了は `exited`（quotaはadmissionでも確定）。
`failed / quota / timed_out / cancelled / validation_failed / deferred` は理由を保存する。
追加QA待ち・QA承認はない。ラベルの採用・公開の条件は各taskがreceiptとして返す。

保存先は `campaigns/<campaign>/workers/<work-id>/attempt-<n>/`。
`launch.json`、owner/exit/result receipt、events/stderr、結果コピーを保持する。
`events.jsonl` は処理完了・quota判定用であり、画像の実提示や応答ID usageの
完全な証拠ではない。[telemetry](../telemetry/README.md) が終了したthreadのrolloutを取得し、
modelと応答IDを検証してからadapterへ渡す。adapterは `telemetry_images` で描画記録と
不変画像の対応を返し、`classify_tool` で操作を分類する。`validate` では実提示の証跡と
レビュー判断を照合する。実Codexの結果receiptにはtelemetry参照が必須である。

`WorkerLedger.history(plan, request)` は登録済みattemptを順に読み、launch・exit・
結果receipt・telemetryの対応を検証して返す。retryで置き換わる `worker_tasks.result` に
依存しない。保留済み結果やcancel後の `telemetry.ref.json` も保持し、証拠がないattemptは
明示的な欠測として返す。別attemptの証拠や壊れたhashを欠測として黙って扱わない。
taskの操作受付は[review services](review/README.md) を使える。

合成workerによる試験では、競合、親終了、supervisor死亡と孤児process、PID再利用、
並列枠、quota、timeout、結果改変、preview採用拒否、receipt後の中断を確認する。
`test_command` は明示的なtest layoutでのみ使え、productionでの別コマンドへの
置換は拒否する。実Codex・ball/player adapter接続はそれぞれの受入試験で扱う。
