# メインcheckoutの定期更新

メインcheckoutの `main` だけを、fetchで取得した `origin/main` のcommitへfast-forwardする。
30分ごとのsystemd user timerから実行する。作業用worktreeのpull/rebase、依存環境の同期、
submodule checkoutの同期は行わない。Git共通ディレクトリが異なるlinked worktreeからの更新は拒否する。

## 導入と操作

merge後、メインcheckoutから [spin ops](../ops_control/README.md) を使う。

```bash
spin ops install --dry-run
spin ops install
spin ops status git-sync
spin ops run git-sync
spin ops on git-sync
spin ops logs git-sync --lines 100
spin ops off git-sync
```

`install` はリンク更新とdaemon-reloadだけ。`on` はtimerの永続有効化と即時起動、`run` は
timerの有効化を変えずoneshotだけを実行する。毎時00分・30分（Asia/Tokyo）、精度1秒、
ランダム遅延なし、取り逃した回は次回起動時に1回実行する。`off` はtimerと実行中の更新を停止する。

共通lockを使う既存jobのserviceも更新するため、初回は `install git-sync` だけでなく全unitを
`spin ops install` で更新する。常駐中の旧dispatcherは新コードへ置き換わらないため、管理下の子の
状況を確認し、通常のoff/on手順で切り替える。旧版や起動途中のsupervisorは
`supervisor_not_maintenance_ready` として延期し、未対応の保守signalを送らない。
unit導入・有効化・実ホスト検証は通常のmerge後の運用手順で行う。

実処理の直接実行は次の入口。`--repo` はmainがcheckoutされたメインcheckoutそのものを指定する。

```bash
.venv/bin/python .agents/ops/git_sync/sync.py --repo /path/to/tennis-lab
```

## 更新の契約

呼出し専用の一時refへ `fetch --no-recurse-submodules` し、取得したobject IDを固定する。
`FETCH_HEAD` や `origin/main` が別のfetchで動いても、その固定IDだけで祖先判定と
`merge --ff-only --no-autostash --no-overwrite-ignore` を実行する。一時refは終了時に削除する。
fetch・ref削除を含むGit hookとrecursive submodule更新は無効にする。

以下は `outcome: skipped` と理由を記録し、終了コード75で更新を見送る。自動stash/reset/rebaseは行わない。

| reason | 更新を見送る条件 |
|---|---|
| `not_main_checkout` | linked worktree、または指定先がrepo rootではない |
| `non_main_branch` | main以外、またはdetached HEAD |
| `tracked_changes` | indexまたは通常の追跡ファイルに変更がある。staged gitlink変更も含む |
| `git_operation_in_progress` | merge、rebase/am、cherry-pick/revert、sequencer、bisect、index/HEAD/main refのlock |
| `index_flags` | assume-unchanged/skip-worktreeが変更を隠し得る |
| `diverged` / `local_ahead` | 固定した更新先へfast-forwardできない |
| `untracked_collision` | 未追跡・ignoredファイル/ディレクトリ/symlinkが更新先と衝突する |
| `protected_path_change` | `.venv` または `.claude/worktrees/` を更新先が変更する |
| `submodule_layout_change` | gitlinkの追加・削除・通常ファイルとの変換。checkout保護のため手動確認が必要 |
| `head_changed_during_preflight` | fetchやservice操作の間にHEADが変わった |
| `checkout_jobs_running` | 有限job、対話CLI、supervisorの処理周期、または別の更新が共通lockを保持している。`lock_holders` に入口・PID・modeを記録し、担当の親を判別できる場合は `parent_issue` も記録 |
| `control_operation_running` | spinによるdispatcherのON/OFF、または別の更新が制御lockを保持している |
| `finite_jobs_running` | 登録済み有限service（旧unitを含む）が稼働・起動・停止中 |
| `unguarded_job_units` | managerが保持するserviceの起動入口が共通launcherを使っていない。全unitをinstallし直す |
| `managed_jobs_running` | dispatcherの判定、親/子、起動予約、未確認の終了記録が残っている |
| `resident_transition` | 常駐serviceが起動・停止の途中 |
| `supervisor_not_maintenance_ready` | 稼働PIDのSIGHUP handlerを確認できない。起動中は次回待ち、旧版は通常のoff/onで切り替える |

未追跡の `data/` 等は、更新先と衝突しなければ保持する。ignoredも衝突検査に含め、
Git自身にもignored上書きを拒否させる。作業用worktree・共有 `.venv`・submoduleの実体は変更しない。
submoduleの作業checkoutは固定版より遅れたままでもよい。依存定義の変更は `dependency_changes`、
gitlinkの前後の固定版は `submodule_changes` に記録する。依存同期・submodule更新は人間が別途判断する。

## 運用jobとの排他

[共通checkout lock](../lib/README.md#メインcheckoutと運用jobの排他) を、更新側は排他、
有限jobは起動前から終了まで共有で保持する。更新が先なら新規jobは対象コードをimportする前に待つ。
jobが先なら更新側は待ち続けず、次の30分周期へ延期する。新しい有限jobにも同じlauncherを使う。
対話CLIは[入口側で自動保護](../lib/README.md#対話cliの保護)する。外部待機で共有lockを解放する入口と、
子プロセスのFD方針は同節を参照する。`spin ops status git-sync` の `last_run.lock_holders` と
logsで現在の延期原因を確認できる。
managerの `ExecStart` も照合するため、古いunitが未更新のままでは更新しない。ファイルだけを
差し替え、daemon-reloadが済んでいない場合も同様に延期する。

dispatcherは各処理周期を共有lockで保護する。更新側は排他lockを取得して新規起動を止め、
既存CLIのstatusから判定・親・子・予約・占有枠を確認する。仕事が残れば停止せず延期する。
空ならmain processへSIGHUPを送り、内部enabledを保持した保守停止と終了後のstatusを確認して更新する。
registryの取得順と、排他lockを維持したstatus/stop呼出は
[共通lockの保守契約](../lib/README.md#registryと保守呼出のlock順)に従う。
board webも停止してから更新する。停止前にactiveかつenabledだった常駐serviceだけを再開し、
unitのenabledとdispatcher内部enabledを再確認する。timerの有効化は変更しない。

spinのdispatcher ON/OFFと保守停止・再開は `ops-control.lock` で直列化する。
手動OFFが先なら更新時にOFFを保ち、更新中に要求されたOFFはその後に適用される。
停止前からOFFのserviceや、保守中に無効化されたserviceを再開しない。
user busの不通・state破損・仕事の証拠不足・停止/再開失敗はエラーとし、OFFや成功と推測しない。

## 状態・中断・ログ

`${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/git-sync/` に保存する。

- `status.json`: 最終結果。`spin ops status git-sync` の `last_run` に同じJSONを表示する。
  `outcome`、`reason`、開始/終了時刻、更新前後と取得commit、依存/gitlink差分、拒否の詳細を含む。
  `resume_pending` は常駐serviceの復旧記録が残っているかを示す。
  `checkout_blocked` は更新中/復旧待ちマーカーの有無を示す。
  `skip_streak` は同じcheckoutで連続する `skipped` の回数、最初の見送りからの経過秒、
  直近に取得した `lock_holders` と通知の状態を示す。更新・変更なし・失敗の結果で連続回数をリセットする。
- `runs.jsonl`: 取得、保守停止、更新、再開、終了のイベント。systemd journalにも同じ内容を出力する。
- `resume.json`: 停止した常駐serviceの再開対象。再開の照合が終わるまで保持する。
  更新が強制終了された場合、次回はunitと内部enabledを再確認して復旧する。手動OFFを上書きしない。
  各行の元のmain PIDと `phase`（pending/stop_requested/stopping/stopped/starting/resumed/left_off）を
  atomicに保存する。HUP送信前にstop_requested、停止確認後にstopped、起動要求前にstartingを記録する。
  停止待ちの旧PIDがactiveでも再開成功とみなさず、停止完了と新PIDの起動を確認してから記録を削除する。

終了コードは0（updated/unchanged）、75（明示的なskip）、2（failed）。unitの `SuccessExitStatus=75` により
延期をsystemdのfailedにしない。unit更新前のfailed状態でも `spin ops status git-sync` は
`last_outcome=deferred` と表示し、生の `active_state`・`last_result`・終了statusは保持する。
serviceの最終結果とjobの理由は区別して確認する。再開が失敗した場合は更新済みでもfailedになり、再開記録を残す。
通常の停止signalではfinallyで復旧を試みる。再開記録の破損は手動確認が必要で、削除して成功扱いにしない。
失敗後も、再開前にmain・追跡ファイル・Git操作を再検査する。部分更新やローカル編集が残る場合は
`resume_blocked_reason` を保存してserviceを停止したままにし、人間がcheckoutを修復した後の再実行で復旧する。
更新前にGit共通ディレクトリへ `ops-code.blocked` を作り、mainの整合性を確認してから解除する。
強制終了や部分更新でマーカーが残った間は、ロックが解放されても新しい有限jobと常駐serviceの
importを拒否する。マーカーを手動削除せず、checkoutを修復してgit-syncを再実行する。
復旧先は記録した更新前または更新先commitとし、更新前へ戻した場合は更新先との未追跡衝突も再検査する。
不明な版や壊れたマーカー、記録と異なるHEADは自動解除しない。

### 連続見送りの通知

連続3回の見送りで `consecutive_skips` をsystemd journalと `runs.jsonl` に記録し、
[共通Windows通知](../d_capacity/README.md) でWindows Terminalの通知先へ送る。
30分の定期周期だけなら、初回から約1時間で通知する。通知には回数・経過時間・直近の理由と
最後に取得したlock保持者の入口・PID・判明した親issueを含める。
送信前の状態は `pending` で保存し、しきい値到達をjournalへ記録してから、通知の呼出直前に `uncertain` を保存する。
送信前に中断した場合や未送信を確認できた場合は、次の見送りで再試行する。
送信済みまたは送信結果不明の場合は、その連続見送りの間は1回の送信試行で終える。
送信結果はjournalと `skip_streak.notification` に残す。

## 検証

```bash
.venv/bin/python -m pytest -n 2 tests/e2e/agents_ops/test_git_sync.py \
  tests/e2e/agents_ops/test_interactive_checkout.py tests/e2e/agents_ops/test_checkout_job.py \
  tests/e2e/agents_ops/test_checkout_registry_order.py \
  tests/e2e/agents_ops/test_ops_control.py tests/e2e/agents_ops/test_dispatcher_supervisor.py
```

bare originとclone、一時worktree・実submodule・fake systemd/プロセス・実flockで検証する。
registry競合テストはdispatcher・registry・agent_execを実装のまま使い、systemdと外部agentだけをfakeにする。
実ホストのunit・メインcheckout・ほかの作業worktreeは変更しない。
