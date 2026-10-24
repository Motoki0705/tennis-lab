# Daily CI controller

毎日 **06:00以上・09:00未満（Asia/Tokyo）** だけ、通常mergeを止めてmainを固定し、
`ci.yml` / `knowledge.yml` をdispatchする。成功で早期解除、期限で未復旧のまま解除する。
controllerと独立watchdogはホストの実時計を読む。production CLIに時刻・日付の上書きはない。
Python APIの`clock`差替えは隔離テスト専用。

## 共通cycle API（version 1）

`PYTHONPATH=.agents/ops` で `from daily_ci import CycleAPI, CycleStore`。
controller、collector（#1497）、dispatcher/supervisor（#1498）、既存merge gateが同じAPIを使う。
`CycleAPI(CycleStore("owner/repo"))` の既定stateは
`${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/daily-ci/<casefold repo SHA256>/state.json`。
worktree・親session単位に分割しない。`CycleStore(repo, directory)` はテスト・明示された接続先用。
本番導入は全呼出元で同じXDG_STATE_HOMEと既定directoryを使う。

| API | 契約 |
|---|---|
| `initialize(baseline_sha, workflows=None, integrations=None)` | 初期基準版は必須の完全SHA。disabledで初期化。再初期化・破損・version不明・初期化後のstate欠落は拒否 |
| `snapshot()` | config、全`cycles`、`last_success`、実時計からの`current_cycle_id` / `in_window` / `normal_merge_frozen` |
| `register_repair(cycle_id, pr, head_sha, signatures)` | 当日active cycleに修復PRと完全head SHA・失敗/診断署名を登録。head更新はmerge試行前に明示再登録 |
| `register_worker(cycle_id, worker_id, handle)` | 起動予約**前**にdispatcherの安定IDと実runtime/予約を辿れる非空JSONを登録。同IDのhandle変更・失効後の再登録は拒否 |
| `permit_worker(cycle_id, worker_id=None)` | 予約取得・実起動・継続・書込み前に毎回検査。期限/現在cycle/有効状態を照合し、target/generation/deadlineを返す。過去のpermitを再利用しない |
| `record_result(result)` | collectorがartifactと必須検査を確認した結果を保存。下記の全identityを照合。remote状態だけから成功を作らない |
| `unresolved(cycle_id, records)` | `{signature,evidence,...}` の未復旧集合を置換。分類不能・取得障害も残す。空集合はcollectorが回復を証明した場合だけ |
| `expire()` | 短いstate lockだけで期限失効・worker権限撤回。repo merge lockやremote停止応答を待たない |
| `store.repository_lock()` | freeze→main固定、既存gateの検査→実merge→結果保存をrepo全体で直列化。競合時は明示busy、lockファイルを削除しない |

cycle IDは`daily-ci-YYYY-MM-DD`（JST日付、repoと組にする）。`generations[]` は1始まり。
初回は`kind=initial`、gateで確認した修復merge後は`kind=repair`と**新しいSHA/新規run**。
同SHAの翌日は別cycle。各generationの`requests[workflow]`は`request_id`、state、run_id、attempt、
remote状態、collector observation、stop_stateを保持する。
cycleは開始/期限・cutoff_sha・target_sha・baseline_sha・observed_main_sha・修復PR・worker・
未復旧署名・carryover・errors・end_reason・freeze_released_at・stop_completeを持つ。
`success`、`expired`、`missed`、`disabled`を区別する。外部main更新は`blocked/external_main_changed`とし、
旧結果で成功にせず、修復権限も止める。次の06:00で新mainと残件を再照合する。
`baseline_sha`は最後の成功版（初回は導入時の明示版）。失敗日や日付変更で進めない。

## Workflow契約（#1496）

workflow_dispatchの文字列inputsは`cycle_id`, `generation`, `request_id`, `target_sha`,
`baseline_sha`, `deadline_at`。**run-nameは `daily-ci:${{ inputs.request_id }}`**。
dispatchのrefは`main`。workflowは`target_sha`をcheckoutし、その実SHAと全inputs、
GitHub run ID/attempt/workflow/job/shard、必須検査の実行と結果をmanifest/artifactへ残す。
run.head_shaも対象SHAと一致させる。時刻と期限を実時計で検査し、遅延実行を成功にしない。
CIのjob構成・性能・検査集合は#1390の契約を維持する。

dispatch前に`dispatch_unknown`をfsyncする。POSTは各requestで**一度だけ**。
`return_run_details=true`の応答を利用するが、応答喪失・controller死後はworkflowのrun一覧を
marker/event/main/path/SHAで照合し、run/attemptへ結合する。0件は再送の根拠にしない。
複数run・ページ欠落・取得障害・identity不一致は記録して停止する。
同runのattempt増加は既存observationを無効化し、別attemptの成功を合成しない。
GitHub REST契約: [workflow dispatch](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event)。

collectorの`result`は追加キーなしの以下のJSON。`result`はsuccess/failure/incomplete。
`evidence`はcollectorが検証したmanifest/artifactへの非空JSON参照であり、単なる自己申告の成功ではない。

```json
{"cycle_id":"daily-ci-2026-10-25","generation":1,"workflow":"ci.yml","request_id":"<32 hex>","run_id":123,"attempt":1,"target_sha":"<40 hex>","checkout_sha":"<40 hex>","baseline_sha":"<40 hex>","result":"success","evidence":{"manifest":"<verified artifact reference>"}}
```

保存先run/attempt、target/checkout/baseline、現在generationが一致し、remote完了も確認できる必要がある。
cancelled/skipped/neutral、取得失敗、JUnit/manifest欠落、必須shard欠落をsuccessにしない責務はcollectorにある。
controllerは最終generationの**全workflow**のremote successとcollector success、未復旧集合の解消、
最新main一致を同じrepo lock下で再照合してからsuccessにする。

## Collector / dispatcher adapters

initの`--integrations FILE`は `{"observer":["command","args"],"workers":["command","args"]}`。
各argvへ`--input <private JSON path>`を追加し、shellを介さず短時間実行する。
入力は`{schema_version:1,operation,repository,state_dir,cycle}`。probeのcycleはnull、他は上記snapshotのcycle。
stdoutはJSON1件、診断はstderr。未知schema・非ゼロ終了・timeout・欠測を成功にしない。

| role / operation | stdout / 動作 |
|---|---|
| observer / probe | `{"schema_version":1,"ready":true,"role":"observer"}` |
| workers / probe | `{"schema_version":1,"ready":true,"role":"workers"}` |
| observer / collect | `{"schema_version":1,"results":[result...],"unresolved":[{signature,evidence,...}...]}`。現在cycleを範囲収集し、未検証・欠測を残す |
| workers / reconcile | `{"schema_version":1,"acknowledged":true}`。既存dispatcherへ初回失敗を即時接続。APIへworker/PRを登録し、担当/資源/根拠/正式reviewを維持する。agentをこのadapter内で待たない |
| workers / stop | `{"schema_version":1,"acknowledged":true,"workers":{"id":{"state":"stopped","evidence":{"runtime":"<termination proof>"}}}}`。cycleの未取得予約も撤回し、全登録workerの終了を照合。未確認はstate=stop_unknownと理由付きevidence |

enableは両probe成功を必須とする。未設定・未接続のadapterを黙って成功にしない。
collectorはcontroller所有stateを書き換えずAPIを使う。dispatcherはpermitと失効を既存supervisorへ接続する。
workerへ`DAILY_CI_CYCLE_ID`を渡し、正式gateも同じ値か`--daily-ci-cycle`を明示する。
枠外の古い修復workerは通常merge再開後もgateで拒否する。

## Merge・期限・再起動

通常gateは同repoの共通mutexを取得し、既存review/coverage/HEAD/依存/競合を維持した上で実merge直前に
`merge_reason`を検査する。active枠では登録済み修復PRの一致headと明示cycleだけを通す。
controller未起動でもenabledの06:00〜09:00は通常mergeを止め、09:00以降は固定期限から再開する。
09:00前に送ったmergeの応答が期限後に返れば記録し、再CIや成功にしない。通信中のremote処理を
取り消せたとは推測しない。GitHubの直接push/直接mergeの強制停止には保護設定が必要で、本実装は変更しない。

独立watchdogは期限で先に失効し、その後CI取消とworker停止を要求する。
応答の受理だけで停止確認にせず、runのcompletedとadapterの終了証拠を待つ。
未照合dispatchは遅延runも追跡し、stop_unknownを保持する。期限解除とstop_completeは別状態。
09:00以降の復帰は当日missedを記録するだけでdispatchしない。過去slotを再生しない。
翌日は新cycleを作り、残PRと未復旧根拠をcarryoverに保存する。旧PRは新cycleへ再登録が必要。

## 導入（親がmerge後に実施）

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m daily_ci --repo Motoki0705/tennis-lab init \
  --baseline-sha <導入時に選んだ履歴検査の基準SHA> --integrations /private/daily-ci-integrations.json
systemd-analyze calendar '*-*-* 06:00:00 Asia/Tokyo'
systemd-analyze --user verify .agents/ops/daily_ci/systemd/*
.agents/ops/install_units.sh --dry-run
.agents/ops/install_units.sh
PYTHONPATH=.agents/ops .venv/bin/python -m daily_ci --repo Motoki0705/tennis-lab enable
systemctl --user enable --now tennis-lab-daily-ci.timer tennis-lab-daily-ci-watchdog.timer
systemctl --user list-timers --all 'tennis-lab-daily-ci*'
PYTHONPATH=.agents/ops .venv/bin/python -m daily_ci --repo Motoki0705/tennis-lab status
```

controller timerは06:00の1起点。Persistent復帰もCLI側で時間枠を照合する。watchdogは独立15秒周期と09:00。
有効化前に#1496のworkflow、#1497のcollector、#1498のadapter/supervisor、正式gateの接続を検証する。
`disable`は失効して停止照合し、別途両timerとcontroller serviceを停止する。既存state/lockを削除しない。
テストは`tests/e2e/agents_ops/test_daily_ci.py`と`test_merge_gate.py`。fake時計/API、使い捨てgit、
fake adaptersを使い、実GitHub書込み・agent起動・ホストunit導入をしない。
