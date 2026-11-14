# 作成から336時間のworktree期限削除

メインcheckoutの `.claude/worktrees/` 配下にある、このrepoに登録されたworktreeを、
作成から336時間（14日）以上経過した最初の日次実行で強制削除する。
dirty・未追跡/ignored・未push/未merge・利用中・queue参照・Git操作中・worktree lockは免除理由にしない。
期限内の編集・commit・利用再開で期限を延長しない。個別承認は不要。

```bash
spin worktrees scan
spin worktrees cleanup --dry-run
spin worktrees cleanup
spin ops install worktree-cleanup
spin ops on worktree-cleanup
spin ops off worktree-cleanup
spin ops run worktree-cleanup
spin ops status worktree-cleanup
spin ops logs worktree-cleanup --lines 100
```

timerは毎日 **04:30 Asia/Tokyo**。`Persistent=true` により停止中に逃した発火を補う。
ON/OFF/status等の契約は [ops_control](../ops_control/README.md)。`agents` 集合には含めない。
service停止は子プロセスを含む。中断された削除は次回cleanupで再開する。
unitの導入・有効化はmerge後のホスト検証で行う。

serviceと手動のspin入口は [共通checkout launcher](../lib/README.md#メインcheckoutと運用jobの排他) を使う。
コードのimport前から終了まで共有lockを保持し、git-syncは排他lockを取得できなければ延期する。
git-syncが先ならcleanupはimport・停止・削除の前に待ち、更新と常駐serviceの復旧が終わってから実行する。
scan/dry-runも複数モジュールのimportと初観測記録を更新中のコードから分離するため同じlauncherを使う。
モジュールの直接実行は終了コード75で拒否する。spinを使わない場合は共通lib READMEの有限launcherから
`python -m worktree_cleanup scan` または `python -m worktree_cleanup cleanup [--dry-run]` を起動する。
更新中断の保護マーカーが残る場合も75で拒否する。マーカーを削除せずgit-syncの復旧手順に従う。

## 作成時刻と状態

[inventory.py](inventory.py) が `.git/worktrees/<id>` の開いたディレクトリにLinux `statx` を実行し、
birth timeを取得する。既存worktreeも同じ方法で移行する。取得不能のときだけ削除ツールの初観測時刻を
`source=first_observed` として保存する。mtime/ctime・最古commit・reflogは作成日時に使わない。
最初に記録した根拠と期限を維持し、後日のbirth timeの取得可否で期限をリセットしない。

状態は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/worktree-cleanup/<repo-key>/`。
repo-keyはGit common directoryから算出し、別cloneの状態を分離する。
`worktrees.json` にUTCの作成時刻、根拠、管理ディレクトリのdevice/inode/birth time/UUID、
パス、HEAD、削除phaseをfsync＋atomic renameで保存する。同パスでも管理ディレクトリが別個体なら
新しい記録にする。対象の管理ディレクトリにはwrite-onceの `tennis-lab-expiry-id` をFD相対で
atomicに作り、birth timeが取れずinodeが再利用されてもUUIDの不一致で再作成を識別する。
UUIDは個体の識別だけに使い、作成日時の根拠にはしない。管理ディレクトリと実体のFDを保持し、
gitdirの対応とパスの個体も再照合する。symlink・不正なUUIDは推測で置き換えず失敗とする。
scanで登録消失を観測した記録は再利用せず、同じ管理名・inodeが現れても新しい初観測として扱う。
削除途中の実体の復旧記録は残し、消失した登録への操作権限だけを失効させる。
scan/dry-runは初観測記録・レポートと対象のGit管理UUIDを永続化する。作業内容・Git参照・稼働jobは変更しない。
対象外のworktreeにはUUIDを作らない。UUID導入前の記録は `identity_migration=unsealed_generation` で示す。
birth timeが既知ならその日時を再利用する。不明な旧記録は再作成と区別できないため、UUID付き個体の初観測から
14日とし、旧inodeだけを根拠に早期削除しない。UUID導入後は欠落scanの有無に依存しない。

`report-latest.json` と標準出力のJSONは、`created_at` / `source` / `deadline` / `decided_at`、
`target`（安全な対象範囲）/ `expired` / `selected`、`discarded_changes` を持つ。
変更にはignored・submodule・隠すindex flagも含む。Git操作中などで確認不能ならnullと理由を示す。
変更確認の失敗で期限を免除しない。`results` と追記型 `results.jsonl` はpath・HEAD・判定UTC・
削除成否・エラーを残す。作業ファイルのバックアップは作らない。

`cleanup --dry-run` の `stop_preflight` はrunner記録・registry・現在のprocessと保存済み停止証拠を
読み取り、対象run、記録なし委任、停止準備の問題、旧証拠の復旧予定を表示する。
関係する破損・来歴不明・探索中断は非0で返す。runnerのstatus/stop・signal・停止証拠の修復・registry更新は行わない。
実際の停止・削除の成功はdry-runでは確定できず、非dry-runの結果で確認する。

## 停止と再実行

[lifecycle.py](lifecycle.py) がローカルの親registry・共通ランナー・training queueを連携する。
対象の委任は `stopped` にして新規起動を止め、共通ランナーのstop/statusで
terminal result・`cleanup_complete`・runner/CLI/groupの消滅を確認してから `worker_ended` を記録する。
共通ランナー導入前など、`worker_runs` 自体がない旧委任は新規起動を止めたうえで、
管理job・native子孫の停止と、その委任のworktree内のローカルprocess不在を確認してから終了扱いにする。
`worktree_expiry.termination.source=local_process_absence` と確認UTCを残す。runner証拠や別マシンの
session終了、task成功を推測した記録にはしない。停止失敗中は終了根拠を新たに書かず、次回に再確認する。
dispatcherの未起動予約は既存cancel APIを使い、起動済jobは停止確認後に `phase=ended` とする。
期限削除は担当とjobの状態を更新し、常駐supervisorのunit有効化や内部enabled、git-syncのresume記録は変更しない。
git-syncの保守停止・再開は排他lock内で実行されるため、この停止処理と並行しない。
対象worktreeで動く親は強制解除（`released`、成功完了にはしない）し、dispatcherの自動継続も止める。
保存した来歴は現在のsession/generationに一致する場合だけ担当停止の根拠とする。
handoff時は来歴を移管先へ更新し、移管先のnativeプロセスが確認できない場合は旧来歴を失効させる。
停止未確認は `recovery_required` / `stop_failed` を保持し、削除成功として返さない。

runnerの探索先は [run_records.py](run_records.py) の浅い正規配置（defaultの `logs/*`、
parent CLIの `worker-runs/*/*`、reviewの `agent-runtime`、dispatcher判定、ops-reportのrunner）と、
registryに明示登録されたrun-dir、および対象processの `AGENT_RUNTIME_RUN_DIR` に限定する。
state root全体の再帰走査や、workflow自身の `result.json` の読み込みは行わない。
破損したrunner記録は、registryと独立した `invocation.json` のcwdから対象との関係を分類し、
`stop_diagnostics` に場所・理由・根拠を表示する。対象外と確認できた記録はそのworktreeの削除を止めず、
関係する記録と関係不明の記録は明示失敗にする。processが継承した環境変数だけでrunnerの所属を決めない。
置換済みの旧委任でローカルrun-dirが存在しない参照は `retired_reference` として報告する。
その場所を参照するprocessが対象cwdにいれば再び検査対象にし、記録欠落を失敗として扱う。

同じユーザーのnative Codex/Claudeプロセスも確認する。
停止準備の最初のbaselineを対象探索より前に保存し、親子関係とPID identityを保持する。
baseline取得時にUID・cwdも観測し、探索中に終了した対象外processを来歴不明へ読み替えない。
snapshot内ではPID・start ticks・親PIDを先に採取し、その後にUID・cwdと現在のidentityを調べる。
後段で消滅・読取不能になっても採取済みstatとcwdを捨てない。PIDが別世代なら元のidentityと親子関係を
保持し、後段のUID/cwdは新世代の値かもしれないため所属未確定とする。
親のstat採取後、子のstat読取前に親が消え、子が既に外側の祖先へreparentされる窓も扱う。
消えた親より後に観測し、誕生時刻と祖先から独立性を証明できない候補は、baseline所属だけで除外せず
不明PIDとして保存する。推測で停止せず、その候補の終了を確認できるまで削除を失敗にする。
同じユーザーで対象cwdを確認できるPython/shell等も対象に含める。registry・queue・runnerの停止前に
捕捉を完了し、後段も再実行も最初のbaselineを引き継ぐ。親PIDの消滅だけでは成功にしない。
探索中にcwdを観測できず消えたPIDは、最初の一覧に含まれた子孫も確認不能として保持する。
探索の中断で捕捉が未完了なら、再実行でも失敗を維持する。元の対象を確認できない証拠を自動で捨てず、
プロセスの終了と保存証拠を人間が確認・修復してから再実行する。
旧v2証拠で対象PIDが一度も捕捉されず、探索完了済みで、記録済みの不明PID・その捕捉済み子孫が
すべて終了し、再探索も空なら、その証拠を `.unscoped-v2.json` に理由付きで保存して新しいbaselineへ進む。
対象PIDを捕捉済みの証拠・不明PIDの残存・探索中断には適用しない。新しい証拠はv3として区別する。
native agentはPID・boot ID・start ticksを照合し、pidfdで親子をSIGSTOPする。
全threadの停止後にも一覧を取り直し、新規の子を含む閉包が停止状態に固定されたことを確認してSIGKILLする。
TERM handlerやSIGCONTを実行させず、確認済みPIDは親が変わっても消滅まで追跡する。
新しく現れたPIDが旧親の外側の祖先に引き取られ、元の子か独立した新規jobかを確認できない場合は、
推測して停止せず非0を返す。`native-stops/` に保存した観測・PID identityは再実行でも引き継ぎ、
当該PIDとその後継候補が消滅するまで削除成功にしない。既存の別processから生まれた子は対象に含めない。
停止失敗には `live` / `uncertain` のPID identityを含め、読み取り診断で残存と来歴不明を区別できる。
既存の手動親はHerdrのprivateなsession/process対応も照合する。
自分のagent祖先は停止できないため、該当する直接呼出しは明示失敗し、mainのserviceから再実行する。
任意の外部daemonや他ユーザーのプロセスを制御可能とはみなさない。
training queueは既存のシェル単語decoderで参照を探し、既存cancel CLIを呼び、terminal記録まで照合する。
停止・確認に失敗した場合は非0で返す。失敗は期限延長や恒久免除ではなく、次回も停止・削除を試みる。

[cleanup.py](cleanup.py) は全scan/cleanupを同じflockで排他する。多重起動は終了コード3。
親registryへの期限削除による更新は [registry.py](registry.py) が既存Storeを通して保存する。
分割registryの担当行と索引の間で停止した場合は専用intentを照合して次回に更新を完了する。
intentの保存後に別の更新があれば上書きせず明示失敗する。
期限到達後の削除intentと元のdirectory identityを先に保存し、実体を同じ `.claude/worktrees/` 内の
`.expired-<id>` へrenameする。移動元leafのFDも保持してrename直後の個体を照合する。
別個体なら停止・登録解除をせず、
NOREPLACEで元へ戻す。復旧先が占有されていれば上書きせず `quarantine_conflict` を残し、
再実行は停止より先に復旧する。rename直後の中断も同じ照合で検出する。
後継を復旧して旧個体の消失が確認されたintentは `superseded` とし、削除成功とは記録しない。
正常に隔離した旧個体は、元のパスからの新規起動を遮断して停止を再確認する。
続いて、開いてidentityを確認した旧Git管理ディレクトリの内容だけをfd相対で削除して登録を解除し、
fdベースのrmtreeで隔離実体を削除する。パスから登録を選び直す `git worktree remove` は使用しない。
管理名が同時に再作成されても新登録に触れないよう、空の旧管理ディレクトリ自体は残す。
空ディレクトリは登録worktreeとして列挙されず、Gitの通常のprune/GCで整理される。
中断後は保存phase・実体identity・現在のGit登録を再照合して不足分だけを行う。
他のmissing worktreeを巻き込む全体pruneはしない。

メインcheckout・対象範囲外のworktree・パス途中のsymlinkは削除しない。
実体内のsymlink（共有 `.venv` 等）はリンクだけを削除し、参照先をたどらない。
別の登録worktreeを含む祖先の削除と、途中で別個体になったパスも拒否して理由を記録する。
worktree削除ではbranchを消さない。merge済branchの整理は既存 [cleanup](../cleanup/README.md) の
`apply-auto` をそのまま使う（別timerの次回実行、または明示実行）。

終了コードは0成功、1調査・停止・削除の失敗、2引数エラー、3scan/cleanup同士のlock競合、75launcher拒否。
`--repo` と `--state-dir` は両コマンドに指定できる。時計の差替えはテスト用Python APIでのみ行う。
`--repo` はlauncherのrepoと同じGit共通ディレクトリのcheckoutに限る。別cloneにはそのcloneのlauncherを使う。

## 検証

```bash
.venv/bin/python -m pytest -n 4 tests/e2e/agents_ops/test_worktree_cleanup.py tests/e2e/agents_ops/test_worktree_generation_races.py tests/e2e/agents_ops/test_ops_control.py tests/e2e/agents_ops/test_cleanup.py
```

一時repo・worktree、注入時計、隔離registry、fake runner/queue/processを使用する。
ホストのsystemd・実repoの削除は実行しない。
