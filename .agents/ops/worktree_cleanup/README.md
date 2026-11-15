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
`worktrees.json` のschema v2にUTCの作成時刻、根拠、管理ディレクトリと実体の
filesystem ID・inode・birth time、管理ディレクトリのUUID、
パス、HEAD、削除phaseをfsync＋atomic renameで保存する。同パスでも管理ディレクトリが別個体なら
新しい記録にする。対象の管理ディレクトリにはwrite-onceの `tennis-lab-expiry-id` をFD相対で
atomicに作り、birth timeが取れずinodeが再利用されてもUUIDの不一致で再作成を識別する。
UUIDは個体の識別だけに使い、作成日時の根拠にはしない。管理ディレクトリと実体のFDを保持し、
gitdirの対応とパスの個体も再照合する。symlink・不正なUUIDは推測で置き換えず失敗とする。

filesystem IDは、開いたFDへの `fstatfs` が返すext4のUUID由来 `f_fsid` を使う。
WSLの再起動でblock deviceの並びと `st_dev` が変わっても同じIDで追跡する。
drvfs/9pは複数のWindows volumeへ同じfsidを返す場合があり、安定した個体照合を提供できない。
現在はext4の永続IDを取得できるfilesystemを対象とし、取得失敗・0のID・他のfilesystemは
明示失敗のJSONと非0終了で記録する。実体のbirth timeを確認できない場合も隔離・削除を失敗にする。

[migration.py](migration.py) が旧device形式のactive記録を、管理ディレクトリのinode・保存済みbirth time・既存UUID、
保存パス、管理ディレクトリの `gitdir` と実体の `.git` の双方向登録、実体inodeを照合して移行する。
隔離済みなら保存された `.expired-<id>` の実体を調べる。旧実体にbirth timeが保存されていない形式は、
管理ディレクトリの保存済みbirth timeと双方向登録を根拠に照合し、実体のbirth timeを追加保存する。
移行後も記録ID・隔離パス・作成日時を引き継ぎ、schema v2のstateをatomicに保存してから再開する。
再実行は移行済み記録を再解釈せず、保存の中断では旧stateか新stateのどちらからも再開できる。
一致を確認できない旧intentは `superseded` として閉じ、`filesystem_migration` に理由・旧identityを残す。
現在の登録は新規個体としてbirth time由来の通常の期限判定を受ける。
ただし、旧 `unregistered` 記録の隔離実体が残る場合は削除義務を閉じない。登録解除後は双方向登録による
世代照合ができないため、移行を `blocked` として保存し、cleanup/dry-runは理由付きで失敗する。
実体の消失を確認できた再実行で終了する。同じ元パスへ作られた新規個体の期限判定は独立して行う。
`filesystem_migrations` に移行・終了理由を表示し、終了済みの旧履歴もstateに保持する。
scanで登録消失を観測した記録は再利用せず、同じ管理名・inodeが現れても新しい初観測として扱う。
削除途中の実体の復旧記録は残し、消失した登録への操作権限だけを失効させる。
scan/dry-runは初観測記録・レポートと対象のGit管理UUIDを永続化する。作業内容・Git参照・稼働jobは変更しない。
対象外のworktreeにはUUIDを作らない。UUID導入前でbirth timeも不明な旧記録は再作成と区別できないため、UUID付き個体の初観測から
14日とし、旧inodeだけを根拠に早期削除しない。UUID導入後は欠落scanの有無に依存しない。

`report-latest.json` と標準出力のJSONは、`created_at` / `source` / `deadline` / `decided_at`、
`target`（安全な対象範囲）/ `expired` / `selected`、`discarded_changes` を持つ。
変更にはignored・submodule・隠すindex flagも含む。Git操作中などで確認不能ならnullと理由を示す。
変更確認の失敗で期限を免除しない。`results` と追記型 `results.jsonl` はpath・HEAD・判定UTC・
削除成否・エラーを残す。作業ファイルのバックアップは作らない。

`cleanup --dry-run` の `stop_preflight` はrunner記録・registry・現在のprocessと観測済みPIDを
読み取り、対象run・記録なし委任・`reference_gate`（参照と読取不能の診断）を表示する。
既に隔離されていれば隔離パスを、隔離前なら元パスを調べ、`stage=after_quarantine` / `before_quarantine`
で区別する。関係するrunner記録の破損・来歴不明や、関連PIDの参照確認不能は非0で返す。
関連未観測の `uninspectable` だけでは非0にしない。runnerのstatus/stop・signal・停止記録の修復・
registry更新は行わない。実際の停止・隔離後の参照消失・削除成功はdry-runでは確定できない。

## 停止と再実行

[lifecycle.py](lifecycle.py) がローカルの親registry・共通ランナー・training queueを連携する。
対象の委任は `stopped` にして新規起動を止め、共通ランナーのstop/statusで
terminal result・`cleanup_complete`・runner/CLI/groupの消滅を確認してから `worker_ended` を記録する。
共通ランナー導入前など、`worker_runs` 自体がない旧委任は新規起動を止めたうえで、
管理job・観測済みnative子孫の停止と、その委任のworktreeへのローカル参照不在を確認してから終了扱いにする。
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

[references.py](references.py) が削除の参照ゲートを実装する。隔離後、同じユーザーのprocessと、
UIDにかかわらず関連を観測したprocessについて、`/proc/<pid>/cwd`・`root`・`exe`・`fd/*`・`maps`
（読めない場合は `map_files`）から隔離パス配下への参照を調べる。fdやmmapだけが残る場合も含む。
`(deleted)` 付きの参照も確認し、パスの共通prefixだけでは別worktreeを巻き込まない。
参照processと観測できた子孫を停止してから再走査する。参照が残る場合や、関連processの参照が
確認不能なら非0で返す。Git登録解除・実体削除の直前にも確認する。既に隔離済みの再実行にも同じゲートを適用する。

関連は、停止準備・隔離前後に対象への参照を観測したPIDとその観測済み親子連鎖、および
parent担当・委任、runner、training queue、dispatcherの管理記録から求める。
対象への参照を持たず外へchdirした子も、観測済みなら停止を試み、停止失敗を保持する。
観測できず対象の参照もない孤児は削除ゲートに含めない。ホスト全体で孤児の不在を証明する処理は行わない。
別UIDという理由だけで関連PIDを除外せず、停止権限がなければ失敗にする。世代を特定できない管理PIDは
確認不能の分類には使うが、そのPID番号だけでsignalを送らない。

参照を確認できないprocessは `uninspectable` にPID・comm・読めなかった項目・関連の有無を残す。
`stop_preflight.reference_gate` と実行結果の `reference_checks` で確認できる。
同じUIDでもnon-dumpableな常駐processは存在するため、関連を一度も観測していない確認不能processを
一律に失敗にすると、対象と無関係なserviceがすべての期限削除を恒久的に止めてしまう。
この場合は診断だけに留める。**確認不能processが見えない参照を持つ可能性は残る**。
関連未観測のprocessについて、この検出限界を採用する。管理job・観測済みPIDの停止失敗は削除を止める。

[processes.py](processes.py) はPID・boot ID・start ticksと観測した親子関係を保持する。
stat採取後に終了しても、それまでに観測したcwdや親子関係を捨てない。最初からZ/XのPIDは終了済みとして除外する。
同じユーザーで対象を参照するPython/shell等も、Codex/Claudeという名前に限らず停止対象とする。
観測済みidentityはpidfdで固定し、親子をSIGSTOPし、全thread停止後の再観測でも子孫を捕捉してSIGKILLする。
TERM handlerやSIGCONTは実行させず、既知のidentityはreparent後も終了まで確認する。
自分のagent祖先は停止できないため明示失敗し、mainのserviceから再実行する。
既存の手動親はHerdrのprivateなsession/process対応も照合する。

停止記録は `native-observed/` のv4 JSONで、観測済みidentityの `tracked` だけを保存する。
停止準備・停止後走査・削除直前の確認で新たに観測した参照元と子孫を、判定より前に耐久保存する。
参照読取後に親が終了しても一致と取得済みの親子辺を残す。観測済み子が参照を失ったり確認不能になっても、
停止義務を次の停止処理と再実行へ引き継ぐ。停止成功が確認されるまで記録を削除しない。
旧 `native-stops/` のv1/v2/v3証拠は上書きせず、妥当な `tracked` だけを読み取って引き継ぐ。
旧baseline・祖先・`unconfirmed`・探索未完了flag・v2の復旧判定は削除ゲートに使わない。
旧証拠が壊れて引き継げない場合は診断に残し、現在の参照確認で判定する。
新しい観測済み停止記録の破損は、既知の停止義務を確認できないため明示失敗にする。
任意の外部daemonや他ユーザーのprocessを制御可能とはみなさない。
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
