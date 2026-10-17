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
pool = WorkerPool(layout, plan, (adapter,))
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
  停止要求はシグナル送信前にadapter版・campaign・plan・requestの一致を検証し、
  別の所有者や異なる要求からの停止を拒否する。
- `gpt-6.1-sol` とplanのeffortを指定し、ユーザーconfig・rules・子agent・skills
  読込を無効にする。ツールのnetwork/GPUを無効にし、CPU内部thread数を工程から
  設定する。作業CWDは `attempt/work`、追加write領域もattempt内の宣言済みdirectory
  に限定する。入力、prompt、supervisorのreceiptはworkerのwrite領域に含めない。
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
  停止の最終状態更新も同じwork lock内でattemptと保存済みreceiptを再確認する。
  完了と停止・retryが競合しても、確定済みreceiptや後続attemptを古い停止で取り消さない。
- Codex合計と同名工程の並列枠をcampaign横断で共有する。異なる上限が併存する場合は
  稼働中planと新規planの小さい上限を適用する。終了済みattemptは枠を返し、結果の
  validationはそのcampaignのadapterで再開する。PSS・保存量・GPU予算によるadmissionは
  後続のresources層で接続する。

標準状態は `ready → running → complete`、未回収終了は `exited`（quotaはadmissionでも確定）。
`failed / quota / timed_out / cancelled / validation_failed / deferred` は理由を保存する。
追加QA待ち・QA承認はない。ラベルの採用・公開の条件は各taskがreceiptとして返す。

保存先は `campaigns/<campaign>/workers/<work-id>/attempt-<n>/`。
`launch.json`、owner/exit/result receipt、events/stderr、結果コピーを保持する。
`events.jsonl` は処理完了・quota判定用であり、画像の実提示や応答ID usageの
完全な証拠ではない。rollout由来の詳細計測はtelemetry層が担当する。

合成workerによる試験では、競合、親終了、supervisor死亡と孤児process、PID再利用、
並列枠、quota、timeout、結果改変、preview採用拒否、receipt後の中断を確認する。
`test_command` は明示的なtest layoutでのみ使え、productionでの別コマンドへの
置換は拒否する。実Codex・ball/player adapter接続はそれぞれの受入試験で扱う。
