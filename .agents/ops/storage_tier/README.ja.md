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
| メイン `ckpt/` | 全体を常にD:へ取得 | 照合済みミラー | 消さない |
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
payloadは再hashする。D:の追加読取りとentryごとのjournal/directory fsyncが増える。全件削除の時間は第2段で測る。
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
読取専用 `index --dry-run` はプロセス検査のエラーを報告するが、manifestの認証や削除許可は行わず、選択した小単位を読める。
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
未完了transactionがあれば復旧まで次の操作を止める。E:のexchange後にmanifest公開で止まった場合は
旧ミラーをstage名で保持する。両treeを調べ、D:/E:全件一致の場合だけ明示的な `verify` で証拠を作り、
確認までは旧コピーを保持する。stageの残骸を黙って消さない。

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
   必要な版/runを選び、`ckpt/` 全体とその単位だけをD:へ戻す。
3. メインoutputs aliasはメインへの参照として分類する。独立した旧E:`.claude/worktrees/<name>/outputs` はconsumer停止中に
   E:`.worktree-outputs/<name>` へrootをrenameする。scanの `legacy_outputs` が対応を示す。
   移行先があれば上書き・結合せず明示比較する。旧manifestのpath・ctimeは再利用せず、新しい正規pathでindexする。
   このrenameは第2段の変更操作である。安定したE:単位を明示して `index` する。
   432GiBを仮に80〜150MiB/sで逐次読込みできれば1pass約49〜92分。
   小ファイルのseekで長くなる。これは計画上の仮定で、全件実測ではない。
   indexはE:1pass、hydrateはさらにE:のhash・コピーとD:の検証が必要。小さな温かいprobeからHDD速度を推定しない。
4. 元のsymlink entryを移行記録へ保存し、**リンク自体**をrenameして退避する。E:の実体は動かさない。
   D:の `data/`・`outputs/` は空の実dirを作る。`ckpt/` はhydrateが全体を公開するまで不在にする。
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
