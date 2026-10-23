# 自律親の開発workflow

入口は [dispatcher bootstrap](../dispatcher/parent_bootstrap.md)。同じworkflowを手動親から使う際も、
担当・実行根拠・子・PR・worktreeと受入条件を先に照合する。人間合意と自律根拠は別のまま扱い、
通常の実装判断は親が具体化する。結果を左右する未決事項・停止指示・情報不足は理由付きで待機する。
これは親の判断手順であり、timerやsupervisorがレビューを起動する仕組みではない。

## 引受・分割・起動

1. bootstrapのaccept後、parentの `status` / `reconcile`、dispatcherの `status` とGitHubの全コメント・正式な親子・
   依存・既存PR・ローカルworktreeを照合する。`session.agreed()` / `authorization()` が有効なこと、
   外部依存の実状態・採用状態が変わっていないことを確認する。根拠を作り直して明示停止を解除しない。
2. 分割が必要な場合だけissues規約で親へ移行し、`create-child` へ共通specを渡す。
   正式な子はその親へ戻し、単独issueは自身の番号を以下の `--child` に指定する。新しい階層は作らない。
3. `launch_plan` の起動対象ごとに、メインcheckoutの `.claude/worktrees/` 配下へ専用branch/worktreeを作る。
   [stacked branch手順](README.md#stacked-branchの確認とrebase) と固定した依存headを使う。
   `start-worker` でdelegate予約とdispatcherのstart-childを接続する。実起動はsupervisorが行う。
   先に起動した子の完了を待たず、設定枠内で独立した全対象を起動する。

共通引数はサブコマンドの前。`parent_state` / `dispatcher_state` はbootstrap入力の値を使い、
手動でも別registryを黙って作らない。parentの `--state-dir` 省略時だけ従来のXDG registryを使う。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo owner/repo --parent 123 \
  --state-dir "$parent_state" --dispatcher-state-dir "$dispatcher_state" \
  start-worker --child 124 --branch codex/parser --worktree "$worker_tree" --prompt-file tmp/task.md
# 必要な場合だけ末尾へ --model gpt-6-astra を付ける。既定の正本はlib/agents.conf。
```

出力は `child/delegation_id/job_id/result=reserved`。起動の成功や成果物の合格ではない。
担当issueの原文・親の具体指示・共有pytest lockと [worker prompt](worker_prompt.md) を共通ランナーへ渡す。
親は子のscope・対象外・受入条件・依存・テスト条件・PR作成時点を指示ファイルへ書く。
モデル選択の理由もissueへの `note` に残す。GPU・Blenderはtraining queueを使い、未指定の資源予算を作らない。

予約後の障害は `reconcile` と保存jobを照合する。delegateだけ保存された場合は、同じdelegation_idで
[dispatcher start-child](../dispatcher/README.md#親workflowへのbootstrap契約) へ接続する。
新しい予約・別sessionで迂回しない。既存の手動worker/PRにmanaged runtimeがなければ、その担当へ戻すか
理由と明示的な引継ぎ条件を記録する。既存PRから新しいworker sessionを自動生成しない。

## 結果確認・修正・部分停止

同じparent共通引数で `inspect-worker --child N` を呼び、最新job・終了理由・設定・最終文のprivateな参照先を読む。
runnerのexit 0も `result=completed` もCLIターンの終了であり、常に `task_success=null`。
dispatcherが終了・回収・worker lifetime lockを照合して現在のdelegationへ `worker_ended` を記録するまで枠を保持する。
親は実diff、受入条件ごとの証拠、テストコマンドと結果、PRのclosing指示・branch/HEADを自分で確認し、
`record-result --child N --evidence-file tmp/tests.md` へ記録する。子の最終文の転記だけで合格にしない。

修正は終了照合後、同じworkerへ戻す。PRが既にある場合も新しいsession・worktreeを作らない。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo owner/repo --parent 123 \
  --state-dir "$parent_state" --dispatcher-state-dir "$dispatcher_state" \
  continue-worker --child 124 --prompt-file tmp/fix.md --evidence-file tmp/continuation.md
```

continuation証拠には終了・既存PR/HEAD・差分確認・残る指摘・今回の修正範囲を書く。
同じbranch/worktree・現在の根拠/要件・正式関係・採用・停止・上限を再検査して新しいdelegation_idを予約し、
dispatcherが前の最新runから同じsessionをresumeする。モデル省略時は前回の選択を保持し、変更指定は拒否する。
旧runtimeの終了未確認・session欠落・設定不一致を新規sessionへ置換しない。

`stop-worker --child N --spec tmp/stop.json` はparentのstopと同じ5項目specを使い、子への詳細・親進捗を記録してから、
その子のruntimeだけへ停止要求する。親全体を止めず、依存子への保留と独立子の起動可能性を `launch_plan` で確認する。
停止要求は終了証拠ではない。supervisorの照合後、解除証拠をparentの `resume --child N --spec ...` へ記録してから
`continue-worker` を使う。根拠の鮮度不一致は明示再判断/再合意へ戻す。

CI待ち・情報不足・権限不足はwaiting、runtime/検証の失敗はfailedとして、理由・事実・影響・再開条件・次の対応を残す。
子が追加の修正で進展しない、対象外へ広がる、同じ失敗を繰り返す場合も待機へ移し、
回数・費用・GPU時間等の未指定の上限を勝手に生成しない。修正・再レビュー・別sessionを無制限に繰り返さない。

## 親単位レビューと順次merge

親がその時点の全open対象PRを固定して明示要求する。部分集合・単独PRの自動レビューは作らない。
同じprivate state、親のCLI identityとghアカウントで既存入口を呼ぶ。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo owner/repo --parent 123 \
  --state-dir "$parent_state" review-prepare --prs 200 201 > tmp/targets.json
# targets.jsonのreview_run_idを読み、親が受入条件・統合上の確認点を指示する
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo owner/repo --parent 123 \
  --state-dir "$parent_state" review-request --run-id "$review_run" \
  --instruction-file tmp/review.md > tmp/review-request.json
.venv/bin/python .agents/ops/review/parent_review.py --state-dir "$parent_state" \
  run --request tmp/review-request.json
```

結果受領を `review-state --state result_received` に記録する。阻害指摘は同じworkerへ戻し、修正diff・最新HEAD・
CI・回帰テストと指摘解消を親が確認して既存の `parent_review.py resolve`、`review-state --state fixes_verified` に残す。
指摘がなくても親の確認は必要。coverage不成立・部分結果・失敗を成功にせず、自動再レビューしない。
重大な対象変更は停止し、必要なら親が理由と新runを明示する。詳細は [review契約](../review/README.md)。

```bash
.venv/bin/python .agents/ops/review/merge_gate.py --repo owner/repo --parent 123 \
  --state-dir "$parent_state" merge
```

既定のdry-runを読み、呼出元環境の `AGENT_AUTO_MERGE` に従う。workflow自身が有効値を設定しない。
共有基盤の許可は実際の人間合意がある場合だけ既存の `shared-permission` に記録する。
自律根拠・採用チェックから許可を作らない。gateはcoverage/HEAD/CI/依存/共有基盤承認と、
親・対象子に含まれる全採用提案の鮮度を検査し、通るPRだけ順次mergeする。
応答喪失・試行済みheadはremoteと試行journalを照合し、記録を消して再mergeしない。

## 待機・完了・担当終了

自由文は `note`、親進捗は `sync` を使う。最終イベントは同じstateで `workflow-report --spec tmp/report.json` を呼んで得る。
specは追加キーなしの次のJSON（evidenceも非空）。CLIの最終JSONはdispatcherの4キー契約を維持する。

```json
{"schema_version":1,"state":"waiting","reason":"CI待ち","resume_condition":"CI完了を確認しdispatcher resumeを明示する","evidence":"対象PRと確認済みHEAD・不足check・次の対応への参照"}
```

completedのevidenceには全体の受入条件ごとの親の確認・実merge/close・残した改善候補を書く。
CLIは現在の根拠/要件・記録済み検証・全対象issueのclosed/completed・open PRの不在・
実mergeのHEAD/branch/repo・gateの試行journal・公開/ローカルの親レビュー・子の終了を照合する。
dry-run結果、子の成功JSON、PR作成やチェック数だけではcompletedを返さない。
親に子がある場合はsyncで進捗を照合するが、未closeの親テーマを勝手にcloseしない。
自律親は生存中にreleaseせず、最終イベント受領後にsupervisorが担当をendedへ移す。
waiting/failedは担当を保持し、子終了イベントまたは理由付きの明示dispatcher resumeで同じ親sessionへ戻す。
手動親は全体受入を確認後、従来の `release --ended` を使う。

## Projectへの接続予定

Project機能のmerge後、担当親が次の節目でparent CLIの `project-sync --event EVENT --spec FILE` を明示呼出しする。
このcheckoutはProjectコードに依存せず、呼出しはまだ行わない。spec・apply・表示項目の正本はProject機能のREADME。

| workflowの節目 | event | 確認する事実 |
|---|---|---|
| 引受・委任予約後 | start | 担当・起動対象・次の作業 |
| 親がPRと検証結果を受領 | pr | PR作成と親の受入待ちを区別 |
| 子停止・CI待ち・権限/情報不足 | stop | 作業段階を保持して待ち理由・再開条件を表示 |
| 解除証拠を記録し継続 | resume | 解消した待ち理由と次の作業 |
| 実mergeと子close確認 | merge | PR merge・子完了を親受入と分離 |
| 全体受入・workflow-report completed | accept | 親が確認した受入証拠 |

更新責任は担当親。定期PMは読み取り専用を維持し、Projectを別の担当台帳にしない。
