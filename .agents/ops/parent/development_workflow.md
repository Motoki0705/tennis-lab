# 自律親の開発workflow

入口は [dispatcher bootstrap](../dispatcher/parent_bootstrap.md)。同じworkflowを手動親から使う際も、
担当・実行根拠・子・PR・worktreeと受入条件を先に照合する。人間合意と自律根拠は別のまま扱い、
通常の実装判断は親が具体化する。結果を左右する未決事項・停止指示・情報不足は理由付きで待機する。
これは親の判断手順であり、timerやsupervisorがレビューを起動する仕組みではない。
worker操作CLIはdispatcherで受理済みの親が使う。手動親の起動は [delegateの契約](README.md#子の作成委任再開) に従う。

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
予約は起動許可の確定ではない。supervisorがscheduleの場所をworkerへ渡し、workerは同じregistry lock内で
担当generation・委任ID・親/子の停止・要件・採用・入力鮮度を再検査してから共通ランナーを呼ぶ。
起動直前の変更は `runtime_cancelled` と理由を残して取消し、予定外の判定agentや別sessionで回避しない。

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
未起動の取消は `cancelled_before_start` として実行済みrunから区別する。修正turnの予約だけを取り消した場合、
`continue-worker` は保存された元runをたどり、元のsession・モデルから再開する。
一度もCLIを起動していないことを全jobの取消記録と起動証拠の不在で確認でき、PRもない場合だけ、
同じbranch/worktree・モデルで明示的に再予約して初回起動できる。起動証拠や再開元が曖昧なら復旧待ちにする。
runner喪失後は [共通ランナーの回収確認](../lib/README.md#状態確認停止中断からの復旧) に従い、
全子孫の回収根拠を `stop --cleanup-confirmation` へ渡してからsupervisorで終了を照合する。
`stop-worker` の停止要求やPID/CLI groupの不在だけでは `cleanup_complete` を補わず、予約枠を保持する。

`stop-worker --child N --spec tmp/stop.json` はparentのstopと同じ5項目specを使い、子への詳細・親進捗を記録してから、
停止した子へruntimeの停止を要求する。親全体を止めず、依存子への保留と独立子の起動可能性を `launch_plan` で確認する。
保留になった依存子にも、起動済みruntimeへの停止要求または未起動予約の取消を伝える。推移的な依存も対象とし、
独立子は継続する。停止要求後もsupervisorが終了を照合するまで実行枠を保持する。
停止要求は終了証拠ではない。supervisorの照合後、解除証拠をparentの `resume --child N --spec ...` へ記録してから
`continue-worker` を使う。根拠の鮮度不一致は明示再判断/再合意へ戻す。
単独issue自身の停止・再開で根拠の本文が変わった場合も、古い自律判断を流用しない。
予約取消もdispatcher共通の検査で行い、既に起動証拠がある予約は解放しない。
`inspect-worker` は起動前取消を明示し、実行結果・最終文を捏造しない。

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

自由文は `note --body-file`、親進捗は `sync` を使う。素の進捗コメントを人間の指示と自動分類しない。
合意後の未分類コメントは [既存の分類手順](README.md#稼働中の親への移行) で確認する。
最終イベントは同じstateで `workflow-report --spec tmp/report.json` を呼んで得る。
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
worker操作とworkflow-reportはdispatcherと同じregistry欠落検査を使う。
scheduleやworker履歴が残っているのにregistryだけが欠落・空になった場合は保存・起動を止め、
[復旧手順](../dispatcher/README.md#registry欠落からの復旧)へ戻す。空stateの作成やinitで迂回しない。

取消（closed/not_planned・duplicate）の直属子は成功数へ加えない。親は元issueの取消理由・残作業の行き先と
workerの終了を確認し、次のspecを `record-cancellation --child N --spec tmp/cancellation.json` へ渡す。
実際の引継ぎ先issueをhandoffへ記載し、残作業がなければその理由を明示する。内容の判断は担当親が行い、
CLIが自由文から引継ぎ先や取消の妥当性を推測しない。

```json
{"schema_version":1,"reason":"旧案を代替案へ統合した","handoff":"残作業は #125 へ引継ぎ済み","evidence":"親が元の範囲と代替案の受入条件・差分、worker終了を照合した"}
```

4キーのみ、各文章は非空。CLIは取消状態・現在の根拠・全jobの終了・open PR不在を検査してから、元issueの
作業状況へ確認内容を追記し、その本文と取消理由をprivate registryの確認記録に結び付ける。
完了確認でも再照合し、記録の欠落・本文/取消理由/根拠の変更・未終了worker・未解決PRは理由付きで拒否する。
正式な親子関係と未チェックの取消行を残す。取消確認だけでは全体のcompletedを作らず、実装完了対象の実mergeと
受入証拠も必要。最終JSONの4キーは維持し、privateなdevelopment_resultには取消を実mergeのprsと別に記録する。

状態を変更するworker操作・取消確認・workflow-reportはmainと同じ親単位の操作ロックを使う。
inspect-workerは読み取り専用。GitHub取得中はregistryロックを解放し、並行更新があれば現在の状態を取得し直すよう
明示エラーで止める。古い予約や確認記録を保存し直さない。正式な上位親を持つ手動親の管理は従来どおり可能で、
誤った親への自律割当の拒否とは区別する。

## Projectへの節目の反映

担当親が以下の節目ごとにparent CLIを明示呼出しする。
Projectの導入・権限・初期登録を [Project README](../project_progress/README.md#導入更新の入口) で確認する。
初期登録前は `--event initial` で正式な親と全子孫を指定し、子の追加時も未登録の全issueをspecへ含める。
未導入・権限不足・更新失敗は理由と次の対応を `note` に残し、Project更新済みと扱わない。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo owner/repo --parent 123 \
  --state-dir "$parent_state" project-sync --event pr --spec tmp/project-pr.json --dry-run
# 計画を照合して同じevent/specに --apply を指定する。省略時はdry-run。
```

`tmp/project-pr.json` の例（PR番号は単一の整数）:

```json
{"schema_version":1,"items":[{"issue":124,"stage":"レビュー中","wait_reason":"","next_action":"親が差分・テストを照合してレビュー対象を固定する","pr":200}]}
```

CLI共通引数はサブコマンドの前、`--event` / `--spec` / `--apply` は後。
各itemは `issue/stage/wait_reason/next_action` を持ち、親の行（単独issueを含む）には `acceptance` も必須。
`pr` はpr/mergeイベントだけに付け、`--prs` や配列は渡さない。specの正本はProject README。

| workflowの節目 | event | specに反映する事実 |
|---|---|---|
| 引受・委任予約後 | start | openの対象を `stage: 作業中`、親の受入は未確認のまま |
| 親がPRと検証結果を受領 | pr | 関連open PRを `pr: 200`、`stage: レビュー中` |
| 子停止・CI待ち・権限/情報不足 | stop | 既存stageを維持し、非空のwait_reasonと次の対応 |
| 解除証拠を記録し継続 | resume | 既存stageを維持し、`wait_reason: ""` で解除 |
| 実mergeと子close確認 | merge | 関連merged PR番号をprへ。closed/completedの子は完了、親は受入待ち |
| 全体受入・最終イベントを返す前 | accept | 親だけ `stage: 完了, acceptance: 受入済み, wait_reason: ""`。全体受入の証拠はworkflow-reportへ |

単独issueのmergeでも親行の受入待ちと全体受入を区別する。子PRだけを閉じるmergeイベントでは、
関連PRのない親をitemsへ追加しない。親の集計はProject側が更新する。
自律親はactiveの間にacceptのProject更新とworkflow-reportを終え、その4キーJSONを最終文にする。
更新責任は担当親。定期PMは読み取り専用を維持する。Projectは担当・実行枠・実mergeの正本にしない。
