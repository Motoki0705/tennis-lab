# 運用ジョブの制御

`spin ops` の実処理と対象表の正本。[targets.py](targets.py) が常駐 service、timer＋oneshot service、
その集合を宣言する。[systemd.py](systemd.py) が user manager と unit の状態を取得し、
[control.py](control.py) が開始・停止手順と照合を実行する。[.spin](../../../.spin/README.md) は引数を渡す入口。

## 操作

```bash
spin ops status                         # 全登録対象（集合は展開・重複除去）
spin ops status agents
spin ops install --dry-run              # bus不要、リンク予定だけ表示
spin ops install                        # 全unitのリンク・更新とdaemon-reloadだけ
spin ops install dispatcher              # 指定対象のunitだけ導入
spin ops on agents
spin ops off agents
spin ops on dispatcher
spin ops off report
spin ops run triage                      # timerの有効化を変えずoneshotを単発実行
spin ops logs dispatcher --lines 100
spin ops logs agents --follow
```

`status` / `install` の対象省略は全対象。`on` / `off` / `run` / `logs` は対象を必須とする。
`run` はtimer＋oneshotだけで、常駐serviceやそれを含む集合は操作前に拒否する。
systemctl と journalctl は常に `--user`。`logs` は全対象unitを `--unit` で指定し、journalctlの終了コードを返す。
user managerが停止していても、過去のjournalは読める。

| 対象 | 役割 |
|---|---|
| `dispatcher` | 常駐supervisorと判定timer/service、既存dispatcher CLIのenable/shutdownを統合 |
| `report` | 運用レポートの予定確認timer/service |
| `triage` | 採用提案のissue化timer/service |
| `agents` | dispatcher・report・triageの集合。同じ個別操作を順番に行う |
| `proposal-expiry` | 改善提案の[期限切れclose](../parent/improvements.md#期限切れcloseとunit導入)を日次実行するtimer/service。agentsに含めない |
| `cleanup` | 既存のmerge済みbranch整理timer/service。agentsに含めない。worktreeの期限削除とは別 |
| `git-sync` | [メインcheckoutの30分周期fast-forward](../git_sync/README.md)。agentsに含めない |
| `worktree-cleanup` | [作成から336時間のworktree強制削除](../worktree_cleanup/README.md)。毎日04:30 Asia/Tokyo。agentsに含めない |
| `storage-tier` | [D:/E:のミラー・取得・未使用単位の削除](../storage_tier/README.md)。通知helperを含む。agentsに含めない |

`on` は `enable --now` を実行し、永続的な `enabled` と `active` を照合する。
dispatcherは先に既存CLIの `enable` を実行する。初回だけ [dispatcherの導入手順](../dispatcher/README.md#systemdへの導入有効化停止)
に従ってstateを初期化する。初期化失敗や状態破損を、disabledへ読み替えない。

`off` はtimerの次回起動を無効化し、実行中serviceも停止する。dispatcherは対象表に宣言した順で
判定timer無効化 → 既存CLIのshutdown → 判定service停止 → supervisor無効化・停止を行う。
`disable` は導入時の直接リンクも削除するため、事前に取得した `FragmentPath` の解決済みソースを
`systemctl --user link` で再リンクする。このリンクは次回起動を有効にしない。
停止の一部に失敗しても後続の停止と集合内の別対象の停止を試み、結果をまとめる。

停止後は全unitの有効化・稼働状態と、dispatcher statusの `enabled`、`active_judgement`、
各親の `children_pending` / jobの `phase` / `occupies_slot` を照合する。
残存・回復待ち・照合不能は `partial_failure` と非0。`shutdown` の成功コードだけでは終了を認めない。
ただしuser managerの接続喪失は対象の `outcome: "unavailable"` とエラーを保持し、後続の停止・照合を続ける。
再リンク元の取得に失敗しても `disable --now` と後続のservice停止を試みる。
接続が復帰して後続対象を停止できても、全体の `manager` / `outcome` は `unavailable`、終了コードは非0とする。
回復待ちの判定や子については [dispatcher README](../dispatcher/README.md) の既存の復旧契約に従う。
終了を確認できない記録を削除・初期化しない。状態が `failed` のserviceは終了済みとして扱い、最終失敗をstatusに残す。

## 導入と接続エラー

merge後にメインcheckoutから導入する。unitはメインcheckoutを参照し、worktreeからのinstallは
リンク作成前に拒否する。[install_units.sh](../install_units.sh) が旧unitの残存・旧sourceの再導入も検査する。
従来の `install_units.sh --enable` は拒否する。導入と有効化は上記の別コマンドで行う。

user managerへの接続を操作前に確認する。bus接続不能・manager未起動・manager情報欠落は
`manager: "unavailable"` と非0で返し、unitのOFF状態を推測しない。操作途中で接続が失われた場合も同じ。
`on` / `run` は接続喪失後に新たな対象を起動しないため、`status` は状態一覧を確定できないため、その時点で中断する。
`install` は導入処理1回とdaemon-reloadの失敗を返し、後続対象の操作ループは持たない。
ログイン状態と `XDG_RUNTIME_DIR` / `DBUS_SESSION_BUS_ADDRESS` を確認し、ログインなしの継続運用には
管理者が `sudo loginctl enable-linger <user>` を実施する。CLI自身はlingerやホスト設定を変更しない。
`install --dry-run` はbusなしでリンク予定を確認できる。

## 状態出力

`status` と `on` / `off` / `run` の照合結果はJSON。全体と対象ごとの `outcome` / `errors` を返す。
`status` は過去のjob失敗だけでは非0にせず、取得・照合の失敗を非0にする。

| unitのfield | 意味 |
|---|---|
| `load_state` | unit読込状態。未導入・読込不能はエラー |
| `unit_file_state` | 次回起動の有効化状態。`enabled`、`disabled`、`linked`、`static`などをそのまま区別 |
| `active_state` / `sub_state` | 現在の稼働状態 |
| `last_result` / `exec_main_status` | serviceの最終結果と終了status |
| `next_elapse_realtime` / `next_elapse_monotonic` | timerの次回予定。それぞれの時計系を保持 |
| `last_trigger` | timerの最終発火 |
| `fragment_path` | user managerが読んだunitソース |
| `main_pid` | 現在のservice main process。常駐supervisorの保守signal対応を照合する |
| `exec_start` | managerが保持する実行入口。checkout更新時に共通launcherの利用を照合する |

適用されない値・空の値はnull。必須propertyの欠落をinactive/disabledで補わない。
reportのtimer予定は予定確認の時刻で、実際のレポート生成slotは
[ops-reportのschedule](../ops-report/README.md) で確認する。

## 対象の追加

timer＋oneshotは `TARGETS` に `timer_target("対象名", "unitのstem")` を1件加え、
`.agents/ops/<job>/systemd/<stem>.{timer,service}` を置く。`status/install/on/off/run/logs` が同じ入口に載る。
常駐serviceだけなら `service_target`、独自手順は `Target` の `before_on` / `off_steps` へ宣言する。
停止順序と残存確認対象の全unitを表に持たせ、CLIに対象名の分岐を追加しない。
集合は `kind="group"` と `members` で定義する。保守ジョブを `agents` へ追加しない。
unitを追加する担当はserviceの停止時に子プロセスも終了する契約を定める。

有限の運用jobは [checkout_job.py](../checkout_job.py) を経由し、対象コードを読む前から
終了まで [checkout lock](../lib/README.md#メインcheckoutと運用jobの排他) を保持する。
`Target.status_file` を指定すると、state rootからの相対パスにあるjob固有JSONを
`last_run` として状態に含める。未実行はnull、読込失敗はエラーとなる。

## 検証

```bash
.venv/bin/python -m pytest -n 2 tests/e2e/agents_ops/test_ops_control.py
```

一時git repoとfake systemctl/journalctl/dispatcher CLIで公開spin入口を実行する。
集合と個別操作の一致、残存unit/子/判定、接続不能・途中切断、リンク復元、導入だけの動作、
対象表＋unitだけの拡張を検証する。実ホストのunit導入・有効化は行わない。
