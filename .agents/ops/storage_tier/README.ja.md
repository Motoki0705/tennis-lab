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
| worktree `outputs/` | D:の実ディレクトリ | `.claude/worktrees/<name>/outputs/...` | task/purpose/experiment/run-id |
| メイン `ckpt/` | 全体を常にD:へ取得 | 照合済みミラー | 消さない |
| worktree `data/`・`ckpt/`・`.venv` | メインへのsymlink | 独立したコピーは作らない | 個別に走査しない |

[config.toml](config.toml) にdatasetの**container**（直下の子が版）と、例外の**単位root**を宣言する。
重なる宣言は拒否し、新しいtaskのnamespaceは設定へ追加する。素材・旧配置の曖昧なものは
`unmanaged` として報告し、再帰的に境界を推測しない。outputsは標準の4階層と既知のpurposeから発見する。
experiment直下にファイルがあれば旧runと区別できないため、その配下をrunとして扱わない。
E:のworktree outputsはGit登録がなくなっても発見する。管理root途中のsymlink、入れ子のdirectory mount、特殊ファイル、
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

`--repo`・`--config`・`--queue-dir` はsubcommandの前に置く。repoはlauncherと同じGit共通dirを持つものに限り、
メインcheckoutはGitから求める。明示引数は、発見した単位か、その単位内のpathを指定できる。
引数なしのmirror/verify/evict/cycleはD:の全単位を選ぶ。index/hydrateは単位の明示指定が必須。
実機probeも明示した単位に限定する。`scan` はpayloadをhashしない。
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
root所有の仮想 `/proc` 直下だけは通常の列挙を使う。ここはdatasetのアクセス履歴ではない。

ミラー・照合前に、fd・cwd・mmapをdevice/inodeで確認する。単位外のhardlink別名も検出する。
プロセスやqueue dirを読めなければ失敗。`jobs/*.job` または `running/*.job` があれば**全単位を保持**する。
Python/Hydra/shellが設定から間接的にpathを決めるため、文字列解析だけで非参照を証明しない。
古いrunning記録はqueue側の手順で解決し、このツールは書き換えない。
最近書かれた単位は設定の静止待ち（既定1時間）を置く。hash・公開の前後のsnapshotで変更も検出する。
hashしたstageと公開後metadataの一致を確認するまで、そのhashをmanifestへ記録しない。

ミラーはD:全件SHA-256→E:の新しいstageへコピー→stage全件SHA-256→tree一致確認。
未記録の既存E:はD:と全件一致するときだけ採用する。記録済みE:の更新も旧metadataが不変の場合に限る。
Linuxの `RENAME_NOREPLACE` / `RENAME_EXCHANGE` で公開し、E:`<mirror-repo>/.storage-tier/manifests/` に
atomic JSONを置く。両側のSHA-256、size/mtime/ctime/device/inodeを記録する。

削除は全regular fileのatimeが14日以上古い場合だけ。空の単位は保持する。
D:を再hashしてmanifestと照合し、E:の全entryと記録したmetadataが不変であることを確認する。
D:を同じ親の一時名へ隔離して再確認してから消す。`ckpt/` は消さない。
照合したroot FDを固定しdevice/inodeを確認して、そのFD相対で削除する。隔離先が別個体へ差し替わったら
削除も元pathへの復旧も行わず、個体不一致を残して手動確認を求める。deviceが同じbind mountも、directoryのmount IDで拒否する。
期限到達済みで証拠がある単位は、日次ミラーで証拠を更新する**前**に削除の照合を行う。
不一致を先に再ミラーして隠さない。取得は既存D:を拒否し、E:のhash、コピー、D:のhashを通して公開する。
取得したファイルのatimeは現在に更新する。

## 記録と復旧

D:`<repo>/.storage-tier/audit.jsonl` に、照合・取得・失敗・削除前確認・完了のUTC時刻、単位、サイズ、
照合方式を残す。変更処理はrepoごとに排他し、dry-runはlockも作らない。同じ場所の `transactions/` に
削除の永続intentを置く。

```bash
"${S[@]}" recover
```

復旧は隔離した単位を上書きせず戻し、**削除を再開しない**。戻し先の占有や個体の変化は明示失敗。
削除途中のtreeは部分削除の可能性付きで戻す。内容を調べ、照合済みのE:から復旧する。
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

メインのdata/outputs/ckptがD:の実dirであることと全計画を検証してから、outputsと共有入力・venvリンクを作る。
既存outputsのsymlinkは拒否し、自動で付け替えない。

既存の [336時間worktree期限削除](../worktree_cleanup/README.md) は、D:の実outputsを含む全体を、
ストレージ照合とは独立して消す。**この配置で両方を有効化する前に、ライフサイクルの統合を合意・実装する必要がある。**
このツールは、別途人間が合意した期限削除規約を弱めたり、timerを無断で止めたりしない。

移行・プロセス検査権限・期限削除との整合を解決した後に導入する。

```bash
spin ops install storage-tier --dry-run
spin ops install storage-tier
spin ops on storage-tier
spin ops status storage-tier
spin ops logs storage-tier --lines 100
spin ops off storage-tier
```

user timerは毎日05:30 Asia/Tokyo、最大15分のjitter。失敗時だけ `OnFailure` からbusctlで
user busの `org.freedesktop.Notifications.Notify` を呼ぶ。成功・保持だけなら通知しない。
ホスト側に通知serviceが必要で、配送失敗はhelper unitの失敗として見える。
有効化前に通知配送と `/proc` の完全な検査を実機で確認する。
systemのmount unitへの順序依存は張らず、毎回mountを確認する。installはunitのリンクだけで、有効化は別操作。

## 移行と検証

1. producer・queueの新規起動・旧pathを保持するconsumerの停止時間を親同士で調整する。
   自動削除の前にworktree期限削除の競合と、user unitから読めない `/proc` を解決する。
   権限変更・他session停止・特権helper導入は暗黙に実行しない。
2. E:、使用中・queue参照、設定の間接参照、hardlink別名、E:絶対pathのコマンド、既存D:コピーを一覧にする。
   必要な版/runを選び、`ckpt/` 全体とその単位だけをD:へ戻す。
3. 安定したE:単位を明示して `index` する。432GiBを仮に80〜150MiB/sで逐次読込みできれば1pass約49〜92分。
   小ファイルのseekで長くなる。これは計画上の仮定で、全件実測ではない。
   indexはE:1pass、hydrateはさらにE:のhash・コピーとD:の検証が必要。小さな温かいprobeからHDD速度を推定しない。
4. 元のsymlink entryを移行記録へ保存し、**リンク自体**をrenameして退避する。E:の実体は動かさない。
   D:の `data/`・`outputs/` は空の実dirを作る。`ckpt/` はhydrateが全体を公開するまで不在にする。
   対象worktreeのoutputsリンクも退避してD:に実dirを作る。選んだ単位を取得し、内容と代表hardlinkを検証して
   入力診断をactivateする。既存D:コピーは明示比較し、hydrateで結合しない。
5. 稼働・待機コマンドは担当sessionで更新する。読取専用cycle後、使い捨ての単位だけで
   mirror→evict→hydrateを確認する。`statvfs` の空き、記録、consumer読込み、失敗通知を確認してtimerを有効化する。
   外部hardlinkがあれば空きが戻らない場合がある。rollbackはconsumer停止中に保存したsymlink entryを戻すが、
   D:の新規書込みは別に保持する。ルートVHDXの移設・圧縮は対象外。

検証は `tests/integration/agents_ops/test_storage_tier.py` の一時実ファイル・擬似mount/procで行い、
実fdとhardlinkも確認する。CLI/worktreeは `tests/e2e/agents_ops/test_storage_tier_cli.py`、
unit導入は既存ops-control、取得エラーは既存configuration pathテストを拡張する。
