調査結果と、コマンド・検証・戻し方を含む[第2段の実行手順書](https://github.com/Motoki0705/tennis-lab/blob/1b42677cf31e91eeefd2b0d346c4d1c08ef40f37/tmp/storage-tier-phase2-runbook.md)を作成しました。**配置切替・実index・hydrate・実削除・unit操作・sudoは未実施です。**

推奨は、**#1859の実際の終了後に、必要分のindex・配置切替・取得をまとめて行う**順序です。

| 調査項目 | 結果 |
|---|---|
| Windows物理D:の空き | **255.82 GiB**（18:38 JST） |
| WSL内ext4の空き | **486.02 GiB**（18:33 JST）。物理D:とは別 |
| `scan` / `cycle --dry-run` | exit 1。未切替のdata・outputs・ckptの3エラーだけ。unknown 0 |
| E:の管理単位 | 337単位。各サイズは[CSV一覧](https://github.com/Motoki0705/tennis-lab/blob/1b42677cf31e91eeefd2b0d346c4d1c08ef40f37/tmp/storage-tier-phase2-unit-sizes.csv) |
| 管理単位のfd/cwd/mmap参照 | 観測時0単位、検査不明0 |
| #1859の旧outputs | 7プロセスの参照あり。学習終了後もwrapper等の参照を再確認する |
| ckpt全体 | **28.33 GiB**。19子ディレクトリ＋直下ファイル投影 |
| 使用中の未分類素材 | Clip StudioのPID 2779140がMeiji動画4本、**28.83 GiB**を開いている |

CLIは固定したmain版 `e4bfc2e76` で確認しました。storage関連コードはmerge commit `54b452531` と同一です。実際のmain checkoutは旧版かつdirtyなので、所有者による変更の整理・main更新が実行前の前提になります。

小単位のindex dry-runは成功しました。14.17 MBの `ckpt/dpvo` は0.621秒、対象metadata・atimeは不変でした。起動時間とcacheの影響が大きいため、大規模HDD速度の実測値にはしていません。

実行順は次のとおりです。各操作の完全なコマンドは[手順書の実行部分](https://github.com/Motoki0705/tennis-lab/blob/1b42677cf31e91eeefd2b0d346c4d1c08ef40f37/tmp/storage-tier-phase2-runbook.md#L137)にあります。

| 順序 | 操作・確認 |
|---|---|
| 1 | 親が#1859の終了、Clip Studio停止、対象worktree所有者、CI保守時刻を調整。mainを更新 |
| 2 | 取得対象を確定し、物理D:・guest両方の容量を確認。`index ckpt`→19子単位→追加の必要単位を逐次index |
| 3 | queue空・runner jobなしを確認し、人間がrunner helperを更新 |
| 4 | mainの3つのsymlink **entry自体を退避**し、D:実ディレクトリを作成。E:実体は維持 |
| 5 | ckpt全20単位と必要単位をhydrateし、verify。動画は下記判断に従う |
| 6 | 独立した旧E:outputsをE:`.worktree-outputs/<name>`へrenameし、worktreeのリンクをD:永続領域へ変更 |
| 7 | 自作64 MiB単位でhardlink→mirror→削除dry-run→実削除→空き容量測定→hydrate→SHA-256読込みを確認 |
| 8 | `activate`、引数なしの実`cycle`を確認 |
| 9 | mainの`install_units.sh`でstorage service/timerを導入。unit経由の単発成功後、有効化 |
| 10 | 最初のscheduled実行の成功・auditを確認。通常の日次は05:30〜05:45 JST |

初回は全337単位をindexしない案です。計画用に20〜100 MiB/sを仮定すると、ckpt indexは約5〜25分、hydrateは約10〜50分＋D:検証。全約996 GiBのindexは約2.8〜14.2時間です。小さい単位から実測して見積り直します。

worktree別のリンク先・容量・移行方法は[一覧](https://github.com/Motoki0705/tennis-lab/blob/1b42677cf31e91eeefd2b0d346c4d1c08ef40f37/tmp/storage-tier-phase2-runbook.md#L71)に記載しました。特に次を分けています。

- `ball-arch-desktop`は終了・参照解放後に再調査して移行。
- `standalone-1641`はmain outputsへのaliasなので、`issue1641-quota-docs`のsuffixを維持。
- `standalone-1715`の専用D:data実ディレクトリは保持。
- 非標準配置・内部symlinkのある旧出力はE:に保全し、無理にindexしない。

sudo手順は、**queueが空でrunner jobもない保守枠に、人間がmainから実行**します。

```bash
runner_service="$(sudo cat [local path])"
sudo systemctl stop "$runner_service" tennis-lab-training-queue.service
sudo bash scripts/github_actions/install_self_hosted_runner.sh
sudo [local path] start
# installerが再起動するため、移行中は再停止
sudo systemctl stop "$runner_service" tennis-lab-training-queue.service

# 配置切替・取得・受入後
sudo systemctl start tennis-lab-training-queue.service "$runner_service"
```

現在のbindは既にE:を指しています。**現行bindを維持するだけなら、切替前の即時再インストールは不要**です。ただし、インストール済みhelperはrepo pathを使う旧版なので、次回起動でD:をbindしないよう更新してから通常運転へ戻します。保守中はローカルCI・queue受付が停止します。

親に残す判断は、主に**通常hydrate対象外のMeiji動画4本の扱い**です。照合してD:へ限定コピーする案を推奨します。ckptと合わせて57.16 GiBで、全量が新規物理割当になった場合の物理D:残量は約198.65 GiBです。所有者が明示E:pathで使う代案も書きました。

削除受入ではguestとWindows物理D:の空きを別々に測ります。guestだけ増えた場合、物理D:の回収まで成功とは報告せず、未確認点として返します。VHDX圧縮・sparse化等は追加しません。失敗時は新しいD:内容を丸ごと保存して旧リンクを戻し、中断した管理操作は`recover`で照合する手順です。

手順書のshell 16ブロック・Python 8ブロックは構文検査済みです。実行はしていません。branchは`codex/storage-tiering`、HEAD `1b42677cf`、tracked変更・追加commitはありません。

なお、リンク一覧取得の`readlink`で、**7件のoutputs symlinkのatimeを更新してしまいました（保護対象のball-archを含む）**。この点は読取専用の意図を満たせていません。リンク先の付替え・payloadへの書込み・時刻を戻す操作はしていません。