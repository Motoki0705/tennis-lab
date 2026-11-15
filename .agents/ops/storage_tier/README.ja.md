# ローカルのD:/E:ストレージ階層

[English](README.md) · [简体中文](README.zh-CN.md)

ローカルの配置・保持規約の正本。D:はWSLルートのext4（`relatime`）で作業実体を置く。
E:は `/mnt/wsl/tennis-data` のHDD上ext4（`noatime`）で正本のミラーを置く。
[Google Driveの保存方針](../../skills/tennis-drive/references/storage-policy.md)は別に維持する。
タスク内の相対名は [OUTPUTS.md](../../../src/tasks/OUTPUTS.md) に従う。

## 配置と単位の境界

メインcheckoutを `R` とすると、`R/<relative>` に対応するE:の場所は
`/mnt/wsl/tennis-data/<Rの先頭slashを除いたもの>/<relative>`。

| 場所 | D: | E: | 掃除の単位 |
|---|---|---|---|
| メイン `data/` | 実ディレクトリ | 同じ相対path | データセット版 |
| メイン `outputs/` | 実ディレクトリ | 同じ相対path | task/purpose/experiment/run-id |
| worktree `outputs/` | D:`<main>/.worktree-outputs/<name>/` へのlink | `.worktree-outputs/<name>/...` | task/purpose/experiment/run-id |
| メインoutputsへの既存alias | メイン `outputs/` へのlink | メイン `outputs/...` | 独立した単位にしない |
| メイン `ckpt/` | 実コンテナ、子dirとrootファイルを個別取得 | 同じ単位のミラー | 消さない |
| worktree `data/`・`ckpt/`・`.venv` | メインへのsymlink | 独立したコピーは作らない | 個別に走査しない |

[config.toml](config.toml) にdatasetの**container**（直下の子が版）と、例外の**単位root**を宣言する。
重なる宣言は拒否し、新しいtaskのnamespaceは設定へ追加する。素材・旧配置の曖昧なものは
`unmanaged` として報告し、再帰的に境界を推測しない。outputsは標準の4階層と既知のpurposeから発見する。
experiment直下にファイルがあれば旧runと区別できないため、その配下をrunとして扱わない。
永続領域 `.worktree-outputs/<name>/` はGit登録・worktreeが消えても発見する。同じworktree名は同じ永続領域を再利用し、
別領域が必要なら別名を使う。worktreeのoutputsリンク自体は参照であり、独立した単位にしない。
scanはリンク先を読まず参照のpathだけを列挙する。statでリンクを辿るだけでも古いsymlinkのatimeが変わるためである。
storage CLIにはメインoutputsか `.worktree-outputs` の正規pathを指定する。
旧 `.claude/worktrees/<name>/outputs/` の実dirは未管理の移行元として `legacy_outputs` に新pathとの対応を出し、index前に移す。
管理root途中のsymlink、入れ子のdirectory mount、特殊ファイル、
単位内のすべてのsymlinkを拒否する。LinuxのreadlinkはO_PATHのfdを使っても古いsymlinkのatimeを更新するため、
読取専用dry-runを守るためにreadlink前に拒否する。管理payloadは実ファイル/hardlinkを使い、既存リンクは書き換えない。

hardlinkはD:内またはE:内で使い、D:/E:間では使わない。コピーは**単位内**のhardlinkを維持し、
別単位・別tierには独立したinodeを作る。削除するのは対象単位の名前だけ。記録には論理サイズ、
重複を除いた割当量、外部hardlink、回収見積り、filesystem空きの実測差を分けて残す。
他の名前が残ればブロックは解放されない。取得時は新しいD:のinodeを作り、他単位とのhardlinkは再現しない。

## checkpoint単位とCI assets

`ckpt/<トップレベルdir>` ごとを独立した `checkpoint` 単位にする。直下の通常ファイルは
`ckpt` という別の `checkpoint-files` 単位（現在README.md）で、rootファイルの引数もこの投影を選ぶ。
`ckpt` 引数で全子単位を選ぶわけではない。どちらもミラー・照合・取得だけで、未使用削除しない。
E:のckptコンテナinode自体は交換・削除しない。旧コードの全ckpt単位による操作は拒否する。

rootファイルのmanifestがあれば、hydrate/mirror/verify/indexは公開・証拠更新の前に、E:のmetadataと保存済みSHA-256の両方を照合する。
明示indexでも不一致の証拠を上書きしない。旧ckpt全体manifestの移行でも、記録済み直下ファイルの部分集合を先に照合する。

rootファイルは直下の通常ファイルだけをstageでコピー・hashし、1ファイルずつ公開する。
置換前やD:から消えたファイルは、対象tierの `.storage-tier/checkpoint-file-history/<transaction>/` に保持し、
この履歴は自動削除しない。子dirは触らない。公開はファイル単位で原子的で、集合全体では原子的でない。
D:の `checkpoint-file-operations/` journalへ照合済みの予定・コピーentry、旧entry・manifestと進行を永続記録し、未完了なら投影の処理を止める。
writerを止めて、[記録と復旧](#記録と復旧)に従い `recover` する。journalと履歴は保持する。
journalを退避して再indexするだけでは復旧にならない。不一致を根拠に退避inodeを消してはいけない。
rootの時刻は投影のmetadata identityから外し、rootのdevice/inode/modeと全直下ファイルのmetadata/hashは検査する。コンテナの権限はD:/E:それぞれで保持し、
rootファイルのpayloadとしてtier間比較はしない。

旧ckpt全体manifestの移行は、まず `index ckpt` でrootファイルを再indexする。
移行が中断したら `index ckpt` を再実行する。既存の退避manifestは現在の旧manifestと全内容が一致するときだけ受け入れ、
不一致なら両方を保持して失敗する。
旧manifestはE:の `.storage-tier/legacy-manifests/` へ保存される。その後scanに出る各 `ckpt/<dir>` を個別にindexする。
旧treeのhashを子単位へ流用しない。D:には実dirのckptコンテナを作り、rootファイルと全子単位を個別にhydrateする。
E:のckptコンテナはそのまま保つ。

mount参照の検査は、evict、旧ミラーの退役（その公開前検査を含む）、照合済みstage破棄だけに行う。
index/verify/mirror/hydrateの読取りは、同UIDのfd/cwd/mmap検査と前後snapshotを維持し、bind先をstatしない。
監査の `mount_scope.inspection_enabled` へ実施有無を記録する。削除ではread-onlyでもroot一致・内部bindをfail closedにする。
通常dirのindex dry-runは従来どおりquietを省くが、rootファイル投影ではmount検査なしのquietを行う。

runnerのbind元は同じ `Config.load(...).mirror` と設定から導き、repoのdata/ckpt symlinkを解決して決めない。
CIはE:のdata/ckptを読み、ミラー済み単位だけが見える。D:だけの新単位は、ミラー成功後（通常は日次cycle後）に見える。
実行時は設定されたE:のmountpointを必須とし、毎回source mountを読み直し（再起動でdevice番号が変わる場合も扱う）、filesystem内rootとdeviceを照合し、`/mnt/...` という表記をFSROOTと直接比較しない。
既存mountのroot/device不一致や書込可能なmountは拒否する。

第2段のD:切替前に、親がCIとrunner queueの保守時間を調整し、実行中jobの終了を待つ。
動作中queueの停止は学習を終了させるため、先に空にする。他sessionの学習は別に調整する。
mainへ反映後、人間がmain checkoutから以下を行う。

```bash
runner_service="$(sudo cat /opt/actions-runner/.service)"
sudo systemctl stop "$runner_service" tennis-lab-training-queue.service
sudo bash scripts/github_actions/install_self_hosted_runner.sh
sudo /opt/tennis-lab-actions/bin/mount_assets.sh start
# installerはserviceを起動するため、配置切替が終わるまで再停止する。
sudo systemctl stop "$runner_service" tennis-lab-training-queue.service
# 親が合意済みのD:切替・hydrate・読取専用受入を行う。
sudo systemctl start tennis-lab-training-queue.service "$runner_service"
```

installerは/optのhelper/config、system unitのtennis-data-mount.service依存を更新し、daemon-reloadとenable/startを行う。
従来のpackage/runtime導入も行うためdownload時間を見込む。現在のbindは既にE:なのでFSROOT/deviceは変わらず、
検査が通れば切替前のunmountは不要。再導入はrepoがD:になった後の次回起動を修正するもので、稼働中bind自体は交換しない。
この段階は手順の実装のみで、sudo・unit操作は実行しない。

## CLI

実行したい実装があるcheckoutから、dry-runも含めて必ず有限の
[checkout launcher](../lib/README.md#既存cliと新規jobの接続) を使う。

```bash
export PYTHONPATH="$PWD/.agents/ops"
export PYTHONDONTWRITEBYTECODE=1
S=(.venv/bin/python .agents/ops/checkout_job.py --repo "$PWD" -- .venv/bin/python -m storage_tier)
"${S[@]}" scan
"${S[@]}" mirror data/ball_detection/example-v1 --dry-run
"${S[@]}" mirror data/ball_detection/example-v1
"${S[@]}" verify data/ball_detection/example-v1 --dry-run
"${S[@]}" index data/ball_detection/example-v1 --dry-run
"${S[@]}" hydrate data/ball_detection/example-v1
"${S[@]}" evict data/ball_detection/example-v1 --dry-run
"${S[@]}" cycle --dry-run
```

`--repo`・`--config` はsubcommandの前に置く。repoはlauncherと同じGit共通dirを持つものに限り、
メインcheckoutはGitから求める。明示引数は、発見した単位か、その単位内のpathを指定できる。
引数なしのmirror/verify/evict/cycleはD:の全単位を選ぶ。index/hydrateは単位の明示指定が必須。
実機probeも明示した単位に限定する。`scan` はpayloadをhashしない。
未分類のE:`data/<トップレベル>` ごとに、regular fileの論理bytesと、directory・symlinkを含む割当bytesを
`unmanaged_data_sizes_E` へ出す。du相当で、各トップレベル内のinode重複を除き、リンク先・入れ子mountは辿らない。
directoryもO_NOATIMEで読み、計測不能は0bytesとせず明示エラーにする。これらは掃除の対象外のままである。
`index --dry-run` は選んだE:単位だけをhashし、サイズ・ファイル数・tree digest・atime保持を出す。
dry-runはlock・manifest・ディレクトリ・記録・コピーを作らない。
終了コードは0=成功または保持、1=安全条件・照合の失敗、2=構文、75=launcherなし。E:未mountはdry-runでも失敗する。

移行後に `activate` で `.storage-tier/active.json` を作る。単位を明示しない実際の日次 `cycle` はこの有効化を必須とする。
共通の `PathResolver` と非Hydra入力境界が、
D:にないDATA/CHECKPOINT/ARTIFACTを取得CLI付きの明確なエラーにする。
診断のためE:の存在をstatするだけで、payloadは読まず自動取得もしない。未mountも別のエラーで止める。
repo外の任意rootや共通境界を通らないconsumerは従来のmissing-fileエラーのまま。
新しいOUTPUTの作成は許可する。学習入口やqueueコマンドの個別変更はしない。

## 照合と除外条件

`mountpoint`・`findmnt` で異なるext4と所定のatime optionを確認する。E:不在を空のミラーと解釈しない。
データ・ディレクトリ・manifestは `O_NOATIME` で読み、失敗時に通常openへ切り替えない。
仮想 `/proc` 直下とkernel生成metadataはdatasetのアクセス履歴ではないため通常の読取りを使う。
注入した実ディレクトリのテストinventoryはO_NOATIMEで読む。

ミラー・照合前に、fd・cwd・mmapをdevice/inodeで確認する。単位外のhardlink別名も検出する。
`/proc/<pid>/status` の実効UIDが実行ユーザーの実効UIDと同じプロセスだけを検査する。
root・CI等の他UIDは対象外とし、UID別の除外数をJSON出力・監査記録の `process_scope` に残す。
同UIDでも、非rootの実行ユーザーに4つのUID値が一致し、fd/maps/cwdがroot:root所有で、3つともEACCES/EPERM、
観測前後のPID開始時刻・UID・Name・所有者が不変のときだけ、非dumpableの根拠として除外する。
明示的な `Dumpable: 1` は除外を否定し、`CoreDumping: 0` は根拠にしない。
所有者の規則は [proc_pid(5)](https://man7.org/linux/man-pages/man5/proc_pid.5.html) に基づく。
非dumpableの判定では、別の所有者・namespaceの対応で根拠が立たなければ失敗する。
件数・読めるcomm（status Name）・PID・根拠を `process_scope.excluded_nondumpable` に残す。
Z/Xは `Threads: 1` かつtask一覧がそのPIDだけのとき終了済みとして別に数える。
Zのleaderに別threadが残る場合があるためである（[proc_pid_fd(5)](https://man7.org/linux/man-pages/man5/proc_pid_fd.5.html)）。
生存中のPIDのmetadata欠落は無視しない。
非dumpableな同UIDプロセス（Codex CLI本体、sshd、gpg-agent等）は検査しない。
別の `permitted-capabilities` 除外は、同一user namespaceを運用前提とする。
対象の `CapPrm` に `/proc/self/status` の `CapPrm` にないbitがあり、mapsとcwdがどちらもEACCES/EPERMの場合に限る。
fd一覧は読めてもよい。観測前後のPID・UID・能力値の不変も確認する。件数・comm・両方の能力値・namespaceの前提を
`process_scope.excluded_capabilities` に記録する。どちらもプロセス名による許可リストは使わない。
能力集合の根拠は [commoncap](https://github.com/torvalds/linux/blob/v6.6/security/commoncap.c) にある。
データを読む学習・スクリプトはdumpableで検査可能なプロセスで動くことを前提とする。
UID不明、想定外のI/Oエラー、どちらの根拠も立たない読取不能は失敗として止める。
namespace同一性は常に `unverified` と記録する。`/proc/<pid>/ns/user` の読取りにも同じptrace検査がかかるため、
handleは開かない（[kernel実装](https://github.com/torvalds/linux/blob/v6.6/fs/proc/namespaces.c)）。
補助検査として観測前後に対象と検査側の `uid_map` を読む。読めて異なる場合や不正なmetadataは失敗する。
読めない場合は失敗にせずエラー内容を記録する。一致は `matched` と記録するが、namespaceの同一性の証明とはしない。
前後両方の観測を監査記録に残す。
他UIDのconsumerは保護されないため、
storage処理とデータconsumerは同じユーザーで動かす。training queueの記録は一切読まない。
待機・予定jobの入力でも14日未使用なら消え、手動hydrateが必要になり得る。
mtime/ctimeの静止待ちは設けず、atimeと開いている参照で判定する。hash・公開の前後のsnapshotで内容の変化を検出する。
hashしたstageと公開後metadataの一致を確認するまで、そのhashをmanifestへ記録しない。

ミラーはD:全件SHA-256→E:の新しいstageへコピー→stage全件SHA-256→tree一致確認。
未記録の既存E:はD:と全件一致するときだけ採用する。記録済みE:の更新も旧metadataが不変の場合に限る。
Linuxの `RENAME_NOREPLACE` / `RENAME_EXCHANGE` で公開し、E:`<mirror-repo>/.storage-tier/manifests/` に
atomic JSONを置く。両側のSHA-256、size/mtime/ctime/device/inodeを記録する。

削除は全regular fileのatimeが14日以上古い場合だけ。空の単位は保持する。
D:を再hashしてmanifestと照合し、E:の全entryと記録したmetadataが不変であることを確認する。
D:を同じ親の一時名へ隔離して再確認してから消す。`ckpt/` は消さない。
root FDを固定し、hash・atimeを含む照合済み全entryの計画に従う。各entryを同じ親の専用退避領域へ
原子的に移し、移した実体を照合してからunlink/rmdirする。元の名前に再作成されたentryは削除対象にしない。
不一致のentryは上書きせず元へ戻し、戻せなければ退避領域へ保持する。単位内hardlinkのctime/link数の自前変更は追跡し、
payloadは再hashする。regular fileはclaim後の読取専用FDでLinux `F_SETLEASE(F_WRLCK)` を取得してから、
最終identity・metadata・SHA-256を照合する。既存のopen FD、所有者・権限エラー、lease未対応、
`/proc/sys/fs/leases-enable` 無効の場合は消さずに失敗し、leaseなしへ切り替えない。
leaseはinode単位で外部hardlinkの別名にも効く。新しいopenはbreakを要求し、保持中の書込みはブロックされる。
unlink直前に `F_GETLEASE == F_WRLCK` を要求し、unlink直後、およびfsync/journal等を終えた `F_UNLCK` 直前にも確認する。
直前のbreakはentryを保持・復元して失敗、直後のbreakは削除済みの可能性がある事故としてphaseと確定削除を
journal/auditへ残して失敗する。SIGIOはLinux `F_SETOWN_EX` で呼出threadへ向け、そのthreadでblockし、
解除後に回収して元のsignal maskへ戻す。最後の `F_GETLEASE`〜`F_UNLCK` 間の通知も失敗へ伝播し、
読み捨てて成功扱いしない。CLIはlease中、そのthreadのSIGIOを専用に使う。

使用中検査はFDのdevice/inodeを照合し、既存の `O_PATH` FDも検出する。
削除経路では、検査可能な同UIDプロセスのmountinfoも読む。単位のfilesystem内パスはツール自身のnamespaceで求め、
snapshot rootをFD固定し、mount IDに対応する自身のmount root/pointと相対pathを合わせる。
quarantine・retained・stage名にも同じ規則を使う。同じdeviceで、rootが単位そのものまたは**内部**にあるmountだけを
`/proc/<pid>/root` 経由でstatし、単位内inodeと一致すれば使用中とする。別mount namespaceの単一file bindも対象。
祖先rootと外部・兄弟rootはstatせず除外する。`data/a` と `data/ab` はcomponent境界で区別する。
`process_scope.mount_scope` にfilesystem内の単位path、検査数、祖先除外数、外部除外数を監査記録する（processごとの観測数）。
読取・権限エラーを空一覧にしない。`//deleted` 等の判定不能rootと、対象内mountpointの到達・読取不能は失敗にする。
対象内rootのstat結果が単位のinodeと一致しない場合もunknownにする。上層mountが下層のfile bindを隠す場合があるためで、
祖先・単位外rootの除外は維持する。
CI用read-only data viewのような、到達不能な祖先/外部bindは対象内inode参照ではなく、statも失敗扱いもしない。
rootが単位と同一ならread-onlyでも対象内のままとする。
既存のプロセス除外規則は適用し、検査可能な所属プロセスがいないnamespaceの一覧までは取得できない。

WSL2 Linux 6.6で、O_PATH・private file bindのどちらも `nlink=1` でwrite leaseを取得でき、
unlink・解除後に `/proc/self/fd/N` またはbind aliasから同じ `nlink=0` のinodeへ再openして書けることを実測した。
既存参照は事前検査で検出するが、consumerが**使用中検査後にO_PATH/proc-FD/bind参照を新規取得せず**、
専用領域のランダムtoken名も探索・操作しないことを前提とする。検査・lease確認・unlinkは別操作で、
競合区間には照合やfsyncも含むため、数マイクロ秒以内とは保証しない。
新しく保持した参照から解除後に削除済みinodeへ書ける場合、最後の名前がなければその書込みを復旧できない。
検出したbreakは削除成功とせず事故として記録する。再現probeはstorageテストの
`test_opath_reference_is_busy_and_can_reopen_an_unlinked_inode` と
`test_external_file_bind_reference_is_inspected_and_survives_unlink`。
`nlink > 1` なら通常は別名側にpayloadが残るが、外部からその名前も消された場合は残らない。
leaseは同UIDによるディレクトリentry操作の排他ではない。個体・link数照合で検出した置換は拒否するが、
悪意ある名前空間操作まで保証しない。D:/E:の追加読取りとentryごとのjournal/directory fsyncが増える。
全件削除の時間は第2段で測る。
実装は `removal.py`。隔離先が別個体へ差し替わった場合も削除や元pathへの復旧はせず、手動確認を求める。
repo/E:の基準mountから単位rootまで全成分をnofollow FDで開き、同一deviceのbind mountもmount IDで拒否する。
さらにcheckpointディレクトリ群のdevice/inodeを照合し、dataset/runへの実体aliasを拒否する。
通常のファイルhardlinkはdataset側の名前を消してもckptの名前を消さない。mount IDは実行中の証拠とし、bootを跨ぐ個体IDとして保存しない。
期限到達済みで証拠がある単位は、日次ミラーで証拠を更新する**前**に削除の照合を行う。
不一致を先に再ミラーして隠さない。取得は既存D:を拒否し、E:のhash、コピー、D:のhashを通して公開する。
取得したファイルのatimeは現在に更新する。

## 記録と復旧

D:`<repo>/.storage-tier/audit.jsonl` に、照合・取得・失敗・削除前確認・完了のUTC時刻、単位、サイズ、
照合方式を残す。変更処理はrepoごとに排他し、dry-runはlockも作らない。同じ場所の `transactions/` に
削除の永続intentを置く。
他UIDの除外範囲も各監査eventに残す。失敗はauditとserviceのjournalだけに残し、通知は送らない。
通常dirの読取専用 `index --dry-run` は単位quietを省く（rootファイル投影はfd/cwd/mmapを検査）。どちらもE:側mount参照検査の証拠にはならない。
manifestの認証・削除許可も行わない。実機確認は `verify --dry-run` で両側を検査し、E:の `quiet` 通過とmetadata不変を記録する。
配置切替前は、既存D:/E:ペアの読取専用対応付けによる確認と、未切替配置を拒否する通常CLIの結果を分ける。
`removals/<transaction>.jsonl` にclaim/削除のintent、移した個体、照合済みhash、削除完了entryを永続記録する。
停止により完了記録のないintentが残った場合は未確認として扱い、削除済みとは断定しない。

```bash
"${S[@]}" recover
```

復旧は隔離した単位を上書きせず戻し、**削除を再開しない**。戻し先の占有や個体の変化は明示失敗。
削除途中のtreeは部分削除の可能性付きで戻す。退避entryは上書きせず `recovered-assets/` へ移し、
全計画は `recovered-transactions/` に残す。不完全・変更済みtreeや退避payloadがあれば `recovery-required/` に状態を置き、
その単位のmirror/evict/cycleを止める。不完全なD:でE:の完全なミラーを上書きしない。

1. 復旧結果、退避entry、journal、全計画を確認し、E:にない変更をすべて保持する。個体不一致・戻し先占有は手動確認する。
2. 部分treeを管理root外に保持してD:の元pathを空け、完全な単位を `hydrate` する。
   またはD:を明示的に完全復旧し、E:に対して `verify` する。
3. 全件SHA-256のhydrate/verify成功時だけ復旧状態を解除する。退避した変更は明示的に照合・統合するまで残す。
   復旧から削除は再開しない。

削除後に停止していた場合も事実を記録するが、空き容量の実測成功とは称さない。
未完了transactionがあれば復旧まで次の操作を止める。mirror/hydrateの公開中断はwriterを止めて `recover` する。
journalの削除、強制index、stageの破棄で迂回しない。通常dirはstage作成前にD:の `publications/` へ `copying` intentを保存し、コピー照合後に `prepared` にする。
rootファイルは照合済みコピーとファイルごとの進行を `checkpoint-file-operations/` に残す。
復旧は永続計画とsource/destinationの安定した再SHA-256を比較し、照合できた残りの公開だけを再開する。削除は再開しない。
`prepared` の公開再開ではsource変更、予定外のdestination、未照合stageを拒否する。新しい証拠の記録前に、
旧manifestの全内容、journal、復旧時のmanifestをD:の `publication-evidence/` に保存する。
旧dirは削除せず対象tierの `.storage-tier/recovered-publications/` へ移し、rootファイルの旧inodeは
`checkpoint-file-history/` に残す。D:/E:を再hashしてからmanifestを記録し、receiptを完了にする。
完了receiptと履歴は保持する。その後 `verify` し、通常のmirror/hydrate/indexでは保存済み証拠との不一致を引き続き拒否する。
照合済みrootファイルコピーの作成前に止まった場合は、旧destinationの内容が保たれているときだけ中止扱いにし、
新manifestは作らない。未対応の旧形式の未完了journalは手動確認まで保持・停止する。stageの残骸を黙って消さない。

`copying` 中断では旧destinationが保たれていることを確かめ、部分コピー・未照合bytesを含むstageを履歴へ回収して中止扱いにする。
新manifestを作らず、復旧後にmirror/hydrateを再実行する。`complete` receiptでも旧stageが残っていれば復旧対象とする。
`archiving` intentを先に保存し、同じ旧dirを `recovered-publications/` へ保持してから完了にする。
manifest記録後・退役開始前の中断と、旧実装の完了receiptにも適用する。manifestは再認証しない。

retained dir作成からidentity記録までに止まった場合、復旧は現在のidentityを固定・永続記録してから回収する。
これは回収先を一意にするcustodyの観測で、payload削除の照合根拠には使わず、記録にも区別を残す。
記録済みidentityとの不一致は引き続き失敗にする。JSONの原子的置換の途中に残る `.<record>.<uuid>.tmp` は通常ファイルとして保持し、
処理は正規の `.json` recordだけを読む。他の想定外の名前・特殊ファイル・破損した正規recordは失敗にする。
各コマンドはcommit済みの制御JSONを検査し、一時ファイルのpath・bytes・`committed=false` を
`uncommitted_json` に明示する。照合根拠は正規の完全なrecordから得る。
manifest・transaction・receiptの途中書込みは、一時ファイルと以前のcommit済みrecordを保持する。
報告された一時ファイルを確認用に残す。

D:`audit.jsonl`・`removals/<transaction>.jsonl` の末尾が途中で切れた場合、通常コマンドはjournalエラーで失敗する。
書込み側を停止して `recover` を実行する。復旧は元journalのSHA-256と完全なprefixの長さを
D:`.storage-tier/journal-repairs/` のreceiptへ先に永続化し、切れた末尾を含む元ファイル全体を
`journal-history/` に保存してから、改行で完結したJSON object行だけを有効なjournalとして原子的に公開する。
receiptを完了にし、`journal_repaired` と履歴path・末尾bytesをauditとコマンド結果に残す。
再中断時は同じ履歴bytesを検査して `recover` を再実行できる。JSON objectが完全でも末尾改行がなければ退避対象とする。
途中の行や正規JSONの破損は失敗にし、保持した証拠を調査して対応する永続recordを復元してから `recover` を再実行する。
競合する記録も保持する。publication・ckpt operationのjournalは原子的JSON置換を使い、同じcommit済みrecordの検査を受ける。

次の永続化順序を[中断境界テスト](../../../tests/integration/agents_ops/test_storage_tier_crash_boundaries.py)で検査する。
[全境界一覧](../../../tests/support/agents_ops/fixtures/storage-crash-boundaries.json)にはJSON一時ファイルの作成/write/flush/fsync/置換、
コピーchunkとmetadata、書込み用 `os.open` のflag、`os.write`/`pwrite`/`writev`、stream write/flush、
truncate/ftruncate、rename/claim、削除、journal追記、復旧の書込みを含める。
JSON encoderの連続fragmentは代表write境界にまとめ、binaryコピーは最初と次のchunkを捕捉する。
[部分書込みテスト](../../../tests/integration/agents_ops/test_storage_tier_writes.py)で8MiB超のコピー、
大きなJSONの途中書込み、切れた追記、修復根拠の変更を検査し、既存ファイルへの未登録writeを追加するとtrace完全一致が失敗する。
directory fsyncの前後は、直前の変更と同じプロセス停止時の可視状態になる。検証対象はプロセスの終了である。
各中断状態を[復旧経路一覧](../../../tests/support/agents_ops/fixtures/storage-recovery-boundaries.json)へ対応付け、
[復旧境界テスト](../../../tests/integration/agents_ops/test_storage_tier_recovery_boundaries.py)で各経路をさらに中断する。
文書化した修復後に、正規E:のpayload一致と引数なし日次cycleの成功を要求する。

| 操作 | 永続化の順序 |
|---|---|
| ミラー新規・更新 | copy intent → stageコピー → 照合済みcopy intent → 公開rename → manifest → 完了receipt → 旧stage退役 |
| hydrate | copy intent → stageコピー → 照合済みcopy intent → 公開rename → manifest → 完了receipt → 復旧marker解除 |
| evict | prepared transaction → 隔離rename → 移動identity → retained作成 → 照合済み計画 → entryのclaim/削除journal → 削除 → audit → 完了transaction → transaction unlink |
| stage退役・破棄 | 親の公開receipt → prepared削除transaction → retained作成 → 照合済み計画 → entryのclaim/削除 → audit → 完了transaction → unlink |
| ckpt直下ファイル | prepared journal → 履歴/コピー → 公開照合根拠 → 各交換/進行 → 各旧ファイル回収/進行 → manifest → 完了journal → audit |
| JSONL修復 | 元digest/prefix付きreceipt → journal全体の保持 → 完全prefixの原子的公開 → receipt完了 → audit。再中断は保持した元bytesで再開 |
| 旧ckpt index | 記録済み直下ファイル照合 → 旧manifest退避 → root-file manifest公開 → audit。再試行は一致する退避manifestを受け入れる |
| recover | treeの復元・保持 → 回収identity記録 → tree/entry回収 → 旧証拠/計画保存 → 部分復旧markerまたは公開照合 → receipt完了/active transaction解除 |

復旧の途中で止まったら再び `recover` する。報告された履歴は保持する。部分D: treeは上記の完全hydrate/verify手順で修復し、
activate後の引数なし日次 `cycle` を確認する。復旧からentryの削除は再開しない。

`RENAME_EXCHANGE` 後の旧E:実体も、交換前のhash付きsnapshotを計画とし、同じclaim・lease・永続journalの経路でだけ消す。
旧実体に更新・追加があれば保持してmirrorを失敗にする。recoverは旧treeとentryをE:内の
`<mirror-repo>/.storage-tier/recovered-trees/`・`recovered-assets/` へ保存し、計画・journalはD:に残す。
公開済みの新ミラーを巻き戻さない。旧実体の削除中断は復旧マーカーを作るため、E:だけの変更を保持・照合してから
全件verify/hydrateを通す。archiveは自動削除しない。
未公開のmirror/hydrate stageも、完全な照合済みsnapshotがある場合だけ同じ削除経路で片付ける。
不完全・未照合のコピーはまずstage名で保持し、`stage_preserved` をauditへ残す。receiptがあればrecoverが履歴へ回収する。

旧実装の孤立stageには、単位を特定できるreceipt/transactionがない場合がある。recoverは管理rootを再走査し、
このstageが残れば `unclaimed storage artifacts` を示して失敗する。次の手順で明示的に保持する。

1. producerを止め、scan/recoverが示す正確なstage pathを選ぶ。
2. 通常のmount/nofollow検査で、同tierの `<tier-repo>/.storage-tier/manual-recovery/` を作る。
   `filesystem.rename(stage, archive)` で元のstage名のまま原子的に移す。`RENAME_NOREPLACE` のため、宛先が占有されていれば失敗する。
   両pathとdirectoryのdevice/inodeを記録する。
3. 全bytesを確認用に保持する。recover後、元のmirror/hydrate/indexを再実行するか、上記の完全verify・部分tree修復を行い、
   引数なし日次cycleを確認する。孤立stage回収テストでこの手順を固定する。
payloadを未照合のまま再帰削除する経路は使わない。

## worktreeと常駐

配置切替後、`git worktree add` の後に実行する。

```bash
"${S[@]}" prepare-worktree /absolute/main/.claude/worktrees/new-worker --dry-run
"${S[@]}" prepare-worktree /absolute/main/.claude/worktrees/new-worker
```

メインのdata/outputs/ckptがD:の実dirであることを確認し、`<main>/.worktree-outputs/<name>/` と
そこへのoutputsリンク、共有入力・venvリンクを作る。永続領域もdataもD:に置き、hardlinkを成立させる。
正しい既存の永続領域link・メインoutputs aliasは維持する。その他の既存outputsは拒否し、自動で付け替えない。
既存symlinkのある初期化dry-runは、atimeを動かすリンク先の検査を読取り前に拒否する。
初期化済みworktreeの検証には実際のprepare-worktreeを使う。正しいリンク先は維持する。

既存の [336時間worktree期限削除](../worktree_cleanup/README.md) はworktree内のリンクだけを消し、参照先を辿らない。
永続outputsはworktree消失後も残り、storage_tierの照合付き削除だけが管理する。期限削除の規約は変更しない。

移行と同UIDプロセスの検査を確認した後に導入する。

```bash
spin ops install storage-tier --dry-run
spin ops install storage-tier
spin ops on storage-tier
spin ops status storage-tier
spin ops logs storage-tier --lines 100
spin ops off storage-tier
```

user timerは毎日05:30 Asia/Tokyo、最大15分のjitter。失敗はjournalと `audit.jsonl` に残すだけで、
通知helperや `OnFailure` は置かない。有効化前に実行ユーザーのプロセス検査を実機で確認する。他UIDは合意上の対象外である。
systemのmount unitへの順序依存は張らず、毎回mountを確認する。installはunitのリンクだけで、有効化は別操作。

## 移行と検証

1. producer・queueの新規起動・旧pathを保持するconsumerの停止時間を親同士で調整する。
   自動削除の前に、同UIDのdumpableなconsumerを検査できることと、非dumpable/能力差の除外根拠・記録を確認する。
   同一user namespaceの運用前提も確認する。読めたuid_mapの不一致や、根拠を証明できない本検査の読取不能は失敗とする。
   それらを無視せず、権限変更や他session停止を暗黙に実行しない。
2. E:、使用中・queue参照、設定の間接参照、hardlink別名、E:絶対pathのコマンド、既存D:コピーを一覧にする。
   必要な版/runを選び、ckptの全子dir・rootファイル投影と、その単位だけをD:へ戻す。
3. メインoutputs aliasはメインへの参照として分類する。独立した旧E:`.claude/worktrees/<name>/outputs` はconsumer停止中に
   E:`.worktree-outputs/<name>` へrootをrenameする。scanの `legacy_outputs` が対応を示す。
   移行先があれば上書き・結合せず明示比較する。旧manifestのpath・ctimeは再利用せず、新しい正規pathでindexする。
   このrenameは第2段の変更操作である。安定したE:単位を明示して `index` する。
   432GiBを仮に80〜150MiB/sで逐次読込みできれば1pass約49〜92分。
   小ファイルのseekで長くなる。これは計画上の仮定で、全件実測ではない。
   indexはE:1pass、hydrateはさらにE:のhash・コピーとD:の検証が必要。小さな温かいprobeからHDD速度を推定しない。
4. 元のsymlink entryを移行記録へ保存し、**リンク自体**をrenameして退避する。E:の実体は動かさない。
   D:の `data/`・`outputs/` は空の実dirを作る。`ckpt/` も実コンテナを作り、子dirとrootファイルを個別hydrateする。
   対象worktreeのoutputsリンクも退避し、独立outputsはD:の永続領域、メインaliasはD:メインoutputsへリンクする。
   正規pathへ選んだ単位を取得し、内容と代表hardlinkを検証して
   入力診断をactivateする。既存D:コピーは明示比較し、hydrateで結合しない。
5. 稼働・待機コマンドは担当sessionで更新する。読取専用cycle後、使い捨ての単位だけで
   mirror→evict→hydrateを確認する。`statvfs` の空き、記録、consumer読込み、同UID検査、失敗のjournal/auditを確認してtimerを有効化する。
   queue記録と他UIDのプロセスは単位を保護しない。
   外部hardlinkがあれば空きが戻らない場合がある。rollbackはconsumer停止中に保存したsymlink entryを戻すが、
   D:の新規書込みは別に保持する。ルートVHDXの移設・圧縮は対象外。

検証は `tests/integration/agents_ops/test_storage_tier.py` の一時実ファイル・擬似mount/procで行い、
実fdとhardlinkも確認する。CLI/worktreeは `tests/e2e/agents_ops/test_storage_tier_cli.py`、
unit導入は既存ops-control、取得エラーは既存configuration pathテストを拡張する。
