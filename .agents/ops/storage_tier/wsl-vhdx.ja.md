# UbuntuのVHDX保守手順

[English](wsl-vhdx.md) · [简体中文](wsl-vhdx.zh-CN.md) · [ストレージ階層](README.ja.md)

Windowsホストを操作する人が実施時期を決める。`Ubuntu` のWSL 2登録を保ったまま作業し、
完了後は `D:\WSL\Ubuntu\ext4.vhdx` に戻す。登録を所有するWindowsアカウントでPowerShellを開く。
作業用VHDXはWindowsの `E:\`、オフラインバックアップは独立した物理ドライブに置く。
storage_tierのミラーは `/mnt/wsl/tennis-data` にマウントされている。

## 上限と必要な空きを決める

PowerShellでWSLのバージョン、使えるコマンド、登録path、ホストの容量を確認する。この手順はWSL 3.0.5を使う。
バージョンが変わった場合はresizeが縮小に対応することを確認し、移動のたびに登録pathを再確認する。

```powershell
wsl.exe --version
wsl.exe --help
wsl.exe --list --verbose
Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' |
    Get-ItemProperty | Where-Object DistributionName -eq 'Ubuntu' |
    Select-Object DistributionName, BasePath, VhdFileName, Version
Get-Volume -DriveLetter D,E | Select-Object DriveLetter, FileSystem, Size, SizeRemaining
Get-Item -LiteralPath 'D:\WSL\Ubuntu\ext4.vhdx' | Select-Object Length, Attributes
```

Ubuntuで `/` がディスク全体を使うext4であることを確認し、仮想ディスクのサイズ（現状1 TiB）と
guest・ホストの空きをbyte単位で記録する。

```bash
findmnt -nro SOURCE,FSTYPE,OPTIONS --mountpoint /
lsblk -bdno TYPE,SIZE "$(findmnt -nro SOURCE --mountpoint /)"
df -B1 / /mnt/d
```

- ext4の現在の使用量に、管理領域・予約ブロック・今後のguest書込みを加えて上限を決める。
  上限は実測した物理D:（この環境では931.5 GiB）より小さくする。上限までVHDXが増えた場合に備え、
  D:の他の使用量、VHDXの管理領域、watchdogの閾値（既定20 GiB）も差し引く。
  例としてWSL CLIの `800GB` は800 GiBで、物理D:との差はこれらの差引き前で131.5 GiB。
  実施時点の値で計算し直す。
- D:上でそのまま縮める場合は、ext4のブロック再配置中に動的VHDXが旧上限まで増える場合に備える。
  WSLはext4を縮めてからVHDXの上限を下げる。実ファイル長563.78 GiBを基準にすると、
  追加で `1024 − 563.78 + 20 = 480.22 GiB` の空きに加え、割当の管理領域と他の書込み分が必要。
  実測の空き366.23 GiBでは追加分を見込む前でも約114 GiB足りない。trimとcompactで
  イメージから物理D:に空きが戻っても、イメージの実容量と空きの合計が1 TiBと予備容量の合計を下回る。
  この測定値ではE:を作業先にする。
- 元のVHDXのファイル長を `B` とする。Windows E:には縮小前の仮想サイズと `B` の合計に
  割当の管理領域を加えた空きを確保する。再配置中の増加と、失敗したイメージを残したまま元の大きさで
  復元する場合に備える。別の物理ドライブにバックアップ用として `B` と管理領域を確保する。
  今回の実測値での小計はE:が約1.55 TiB、バックアップ用が約564 GiB。
  C:は物理的に別だが空きが約77 GiBなので、容量が足りるドライブを接続してから進める。例では `F:` を使う。
- D:に置くVHDXは一度に1つ。watchdogの閾値に加えて他の書込み分を確保する。
  UbuntuをE:へ移した後、空きが元のバックアップのファイル長とこの余裕分を収容できることを確認する。
  D:へ戻す直前には、縮小後の**Windows上の実測ファイル長**に閾値・他の書込み・割当の余裕を加えた分以上を
  D:に確保する。たとえば550 GiBのファイルなら、追加分を見込む前でも570 GiB以上が必要。
- 実効速度を80〜150 MiB/sと仮定すると、550〜565 GiBの1パスに約1〜2時間かかる。
  バックアップ用ドライブで書込みとhash照合の2パス、E:で往復の移動の2パスを見込む。
  D:側のhash、trim、オフラインのファイルシステム検査、縮小と起動確認を含め、最初の見積りは6〜10時間とする。
  再配置に時間がかかる場合や復元する場合は延びる。
  2026-11-22の復旧で約891 GiBのファイルをDiskPartでcompactした実測は、その工程だけで約18分だった。[障害記録](https://github.com/Motoki0705/tennis-lab/issues/2361)

## Ubuntuの再起動をまたいで自動処理を止める

新規の投入を止め、稼働中のCI・学習・dispatcher・agentのjobを終了まで待つ。
Windowsのタスクスケジューラで `TennisLab-WSL-GitHub-Runner` の有効化状態と実行状態を控え、あれば無効化して終了する。
Docker Desktopを閉じ、Ubuntuを起動し得るWindows側の他の処理も止める。
[Windowsのkeepalive登録スクリプト](../../../scripts/github_actions/register_wsl_keepalive.ps1)

user unitを動かすLinuxユーザーで、次の出力をUbuntuの外に記録する。正確なunit名と有効化・稼働状態、unitファイルの場所を控える。
導入前のwatchdogなど `LoadState=not-found` のunitは未導入と記録する。他の読込み異常やmanagerの接続エラーは解消してから進む。

```bash
runner_service="$(sudo cat /opt/actions-runner/.service)" || exit 1
[[ "$runner_service" =~ ^actions\.runner\.[A-Za-z0-9_.@-]+\.service$ ]] || exit 1
system_units=(tennis-lab-training-queue.service "$runner_service")
user_units=(
    tennis-lab-dispatch-supervisor.service
    tennis-lab-board-web.service
    tennis-lab-d-capacity.service
    tennis-lab-{storage-tier,dispatch-judgement,ops-report,ops-triage,improvement-expire,memory-autodream,cleanup,git-sync,worktree-cleanup}.{timer,service}
)
fields=Id,LoadState,UnitFileState,ActiveState,FragmentPath
sudo systemctl show --property="$fields" "${system_units[@]}"
systemctl --user show --property="$fields" "${user_units[@]}"
systemctl --user list-units --all 'tennis-lab-*.service' 'tennis-lab-*.timer'
```

runnerのunit名はinstallerが参照するファイルから取得する。watchdogは #2372 のuser service。
storage-tierは `Persistent=true` によりtimerの起動時に未実施分を実行できるため、timerとserviceの両方を対象にする。
検証で使うE:のマウントとCIの読取り専用マウントには `tennis-data-mount.service` と `tennis-lab-actions-assets.service` を使う。
[CI installer](../../../scripts/github_actions/install_self_hosted_runner.sh) · [運用対象の定義](../ops_control/targets.py) · [boardのunit](../board/systemd/tennis-lab-board-web.service) · [watchdogのunit](https://github.com/Motoki0705/tennis-lab/blob/7fce1cbce63a262d3c1746a8a3d1b72917e18698/.agents/ops/d_capacity/systemd/tennis-lab-d-capacity.service) · [storage-tier timer](systemd/tennis-lab-storage-tier.timer) · [timerの仕様](https://www.freedesktop.org/software/systemd/man/255/systemd.timer.html#Persistent=)

ローカルqueueは、メインcheckoutで元の `TRAINING_QUEUE_DIR` を使って `status` の出力を控える。
一時user serviceで動かしている場合は、一覧から `ExecStart` に `training_queue.sh serve` を含む実際のunitを特定する。
元の起動方法・引数を記録してから、同じqueueディレクトリに対して `stop` を実行し、実行中のjobとworkerの終了を確認する。
一時worker監視やほかの定期起動も、稼働中であれば終了または一時停止する。
[ローカルqueueの操作](../../skills/training-queue/SKILL.md#subcommands) · [worker監視のunit名](../parent/worker_monitor.py)

次の各ディレクトリに `zz-tennis-lab-vhdx-maintenance.conf` を新規作成し、pathを控える。必要ならディレクトリも作る。
userの設定先は実行ユーザーの `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user`。
同名のファイルが既にあれば、前回の作業状態を確認してから進む。

- 記録した導入済みのsystem unitごとに `/etc/systemd/system/<unit>.d/`。
- user設定先の `tennis-lab-.service.d/` と `tennis-lab-.timer.d/`。systemdが対応する接頭辞のuser unitに適用する。

```ini
[Unit]
ConditionPathExists=!/
```

`/` が存在するため、この条件で起動をskipする。Ubuntuのファイルシステムに置くのでWSLの終了後も残り、元の有効化状態と
installerのリンクを保てる。状態を記録したときと同じshellで両managerに再読込みさせ、timerを停止する。
各コマンドの終了状態を確認する。
[systemdのdrop-inと条件](https://www.freedesktop.org/software/systemd/man/255/systemd.unit.html) · [installerのリンクの扱い](../ops_control/README.md#操作)

```bash
sudo systemctl daemon-reload
systemctl --user daemon-reload
systemctl --user stop 'tennis-lab-*.timer'
```

storage-tierなど実行中のoneshot jobが終了してからserviceを停止する。

```bash
sudo systemctl stop "$runner_service"
sudo systemctl stop tennis-lab-training-queue.service
systemctl --user stop 'tennis-lab-*.service'
```

記録したsystem unitを `systemctl show --property=Id,LoadState,ActiveState,DropInPaths <unit>`、user unitを同じコマンドの
`--user` 付きで確認する。起動可能な導入済みunitが保守用drop-inを読み込み、すべてinactiveであることを確認する。
同じscopeで `systemctl cat <unit>` を使い、読み込んだ設定の条件も確認する。
一時queue・worker監視が停止し、Windowsのkeepaliveも無効であることを確かめる。
保守中はUbuntuを起動する**たびに直後に**照合し、想定外の起動があればtrimや検証より先に解消する。

## バックアップ、trim、compact

1. 上記の停止を確認してWSLターミナルを閉じる。WindowsのPowerShellから全distroを停止し、一覧で停止を確認する。
   以後、コマンドの失敗や登録pathの不一致があれば作業を中断する。

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --list --verbose
   ```

2. Windowsドライブ文字と物理ディスク番号を照合する。バックアップ用には元のイメージ全体を収容できる独立した媒体を使い、
   ディスク番号と実際の接続先を確認する。下の `F` は実際のドライブ文字に置き換える。元のVHDXを移動・変更する前に
   停止中のコピーを完了し、SHA-256を照合する。バックアップのpath、hash、元のACL、Linuxのユーザー名をUbuntuの外に控える。
   D:へ戻した後の検証が終わるまで、このバックアップを保持する。

   ```powershell
   Get-Partition -DriveLetter D,E,F | Select-Object DriveLetter, DiskNumber
   Get-Disk | Select-Object Number, BusType, Size
   Get-Volume -DriveLetter F | Select-Object DriveLetter, FileSystem, SizeRemaining
   ```

   ```powershell
   $ErrorActionPreference = 'Stop'
   $original = 'D:\WSL\Ubuntu\ext4.vhdx'
   $workingDisks = (Get-Partition -DriveLetter D,E).DiskNumber
   $backupDisk = (Get-Partition -DriveLetter F).DiskNumber
   if ($backupDisk -in $workingDisks) { throw 'Use a physically separate backup drive' }
   if ((Get-Volume -DriveLetter F).SizeRemaining -lt (Get-Item -LiteralPath $original).Length) {
       throw 'Insufficient backup space'
   }
   $backupDir = 'F:\TennisLab-Ubuntu-before'
   $backup = Join-Path $backupDir 'ext4.vhdx'
   if (Test-Path -LiteralPath $backupDir) { throw 'Choose a new backup directory' }
   New-Item -ItemType Directory -Path $backupDir
   ```

   新しいディレクトリのアクセスを制限してから、同じPowerShellでコピーし、照合する。

   ```powershell
   Copy-Item -LiteralPath $original -Destination $backup
   $originalHash = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash
   if ((Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash -ne $originalHash) {
       throw 'VHDX backup hash mismatch'
   }
   $originalHash
   (Get-Acl -LiteralPath $original).Sddl
   ```

3. 別媒体のバックアップの照合を終えてから、停止中のUbuntuをE:のアクセス制限付きの新しい作業ディレクトリへ移す。

   ```powershell
   wsl.exe --manage Ubuntu --move 'E:\WSL-maintenance\Ubuntu-work'
   ```

4. 登録の `BasePath` とVHDXがE:の作業先になり、元のVHDXのpathが空いていることを確認する。
   両ドライブで予定どおり空きが確保できたことを測る。
   保守のために `wsl.exe -d Ubuntu` で起動し、自動処理の停止を再確認する。WSLはcontainerと判定されるため、
   `ConditionVirtualization=!container` が付いた `fstrim.timer` はskipされる。
   mount中のルートを手動でtrimし、Ubuntuを抜ける。

   ```bash
   systemd-detect-virt --container
   systemctl cat fstrim.timer
   systemctl show fstrim.timer -p ActiveState -p ConditionResult
   sudo fstrim --verbose /
   exit
   ```

5. WSLを再度終了し、作業用VHDXをcompactする。実行前後のWindows上の `Length` を比較する。
   compactは物理ストレージの割当を回収する。仮想容量の上限は次の工程で変更する。

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --manage Ubuntu --compact
   ```

## ext4とVHDXの上限を縮めてD:へ戻す

1. Ubuntuを停止し、PowerShellで計画したサイズを渡す。WSL 3.0.5の `--resize` はutility VMで
   オフラインのext4ディスクに `e2fsck` と `resize2fs` を実行する。ext4の縮小が成功するとディスクを
   切り離してVHDXの上限を縮める。後半で失敗した場合は、ext4だけが小さくなっていることがある。
   [WSLホスト側の実装](https://github.com/microsoft/WSL/blob/3.0.5/src/windows/service/exe/LxssUserSession.cpp#L1854-L1911) · [utility VM側の実装](https://github.com/microsoft/WSL/blob/3.0.5/src/linux/init/main.cpp#L2940-L2998)

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --manage Ubuntu --resize 800GB
   ```

2. 作業中のUbuntuを起動して自動処理の停止を再確認し、上記Linuxコマンドで容量を確認する。たとえば800 GiBなら `lsblk` に
   `858993459200` bytesと出る。ルートに書込み可能で、guestの空きが十分であることも確認する。
   WSLを終了し、さらに物理容量を回収する必要があれば停止中に `--compact` を再実行する。
   PowerShellで最終的なファイル長とD:の空きを測る。

   ```powershell
   Get-Item -LiteralPath 'E:\WSL-maintenance\Ubuntu-work\ext4.vhdx' | Select-Object Length
   Get-Volume -DriveLetter D | Select-Object Size, SizeRemaining
   ```

3. D:に戻せる空きがあり、元のVHDXのpathが空いていれば移動する。登録pathを確認してUbuntuを起動し、自動処理の停止を再確認する。
   ルートのブロックデバイスサイズ、物理D:の空き、E:のミラーmountを確認する。
   既知の `data/` 入力と `ckpt/` を通常のpathから読み込めることと、通常のworktreeの `outputs` が
   E:を指すことも確認する。確認が済むまで別媒体のバックアップを保持し、その後で通常運転へ戻す。

   ```powershell
   wsl.exe --manage Ubuntu --move 'D:\WSL\Ubuntu'
   ```

## 中断・失敗から戻す

1. PowerShellからWSLを停止し、登録の `BasePath` と両方のVHDXのpathを確認してから次の操作を行う。
   ディスク変更前の移動に失敗し、登録先のイメージが元のSHA-256と一致するなら、そのイメージを使う。
2. 別媒体のバックアップを記録済みのSHA-256と照合する。登録先を使える場合は失敗したイメージや部分コピーを
   退避して元のファイル名を空ける。D:から退避先へコピーする場合は、ファイル長とSHA-256を照合してから
   D:側の失敗ファイルを取り除く。容量と空の宛先を確認して、バックアップを `Copy-Item -LiteralPath` で現在の登録先に
   コピーする。コピーのSHA-256を照合し、登録ユーザーとWSLが使えるownerとACLを確認する。
   E:で復元した場合は元のファイル長でD:の空きを照合してからD:へ戻す。
3. E:が使えず登録先がE:の場合は、元のイメージと予備容量がD:に収まることを確認する。別媒体のバックアップを
   D:の新しいディレクトリへコピーし、上と同様にSHA-256と権限を確認して、空いている復元用の名前で登録する。
   元の登録は調査のため保持する。[WSLのimport-in-place](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#import-a-distribution-in-place)

   ```powershell
   wsl.exe --import-in-place Ubuntu-recovery 'D:\WSL\Ubuntu-recovery\ext4.vhdx'
   ```

4. 復元したdistroを元のLinuxユーザーで起動する。復元名で起動するときは
   `wsl.exe -d Ubuntu-recovery --user <original-user>` を使う。自動処理の停止、元の仮想容量、D:の空きを確認する。
   E:のミラーmountと入力・worktreeの出力pathを復旧・確認してから通常処理を再開する。
   復元名を使う場合は、Windows側の起動元のdistro指定が検証済みの登録を指すまで起動元を止めておく。
   復元後の確認が済むまで別媒体のバックアップを保持する。

## 通常運転へ戻す

D:上での検証を終えたら、復元後も含め、記録した保守用drop-inだけを削除する。
`sudo systemctl daemon-reload` と `systemctl --user daemon-reload` を実行し、`UnitFileState` を保存した一覧に照合する。
元から `systemctl enable --runtime <unit>` などの一時設定を使っていた場合は、元のscopeで戻す。

保守前にactiveだったsystem unitはsudo付きの `systemctl start <unit>`、user unitは `systemctl --user start <unit>` で再開する。
watchdogが稼働していた場合は最初に起動し、`/run/tennis-lab-d-capacity/status.json` に新しい測定結果があることを
CIの受付前に確認する。CI queueの後にrunnerを再開する。ローカルqueueと一時worker監視は記録した起動方法で再開し、
最後にWindowsのタスクスケジューラを元の状態へ戻す。稼働状態を保守前の記録と照合する。
`Persistent=true` のtimerはこの時点で未実行分を起動することがある。
