# UbuntuのVHDX保守手順

[English](wsl-vhdx.md) · [简体中文](wsl-vhdx.zh-CN.md) · [ストレージ階層](README.ja.md)

Windowsホストを操作する人が実施時期を決める。`Ubuntu` のWSL 2登録を保ったまま作業し、
完了後は `D:\WSL\Ubuntu\ext4.vhdx` に戻す。登録を所有するWindowsアカウントでPowerShellを開く。
作業中のVHDXとオフラインバックアップは、Windowsの `E:\` に用意したアクセス制限付きディレクトリに置く。
storage_tierのミラーは別のext4マウント `/mnt/wsl/tennis-data` にある。

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
- 元のVHDXのファイル長を `B` とし、WindowsのE:には `2 × B + 縮小前の仮想サイズ` と
  ファイルシステムおよびVHDXの管理領域を確保する。バックアップ、ブロック再配置中の増加分、
  失敗したイメージを残して復元する場合のコピーを含む。ファイル長が約564 GiBで上限が1 TiBなら小計は約2.1 TiB。
- D:に置くVHDXは一度に1つ。watchdogの閾値に加えて他の書込み分を確保する。
  UbuntuをE:へ移した後、空きが元のバックアップのファイル長とこの余裕分を収容できることを確認する。
  D:へ戻す直前には、縮小後の**Windows上の実測ファイル長**に閾値・他の書込み・割当の余裕を加えた分以上を
  D:に確保する。たとえば550 GiBのファイルなら、追加分を見込む前でも570 GiB以上が必要。
- HDDの実効速度を80〜150 MiB/sと仮定すると、550〜565 GiBの1パスに約1〜2時間かかる。
  バックアップ・そのhash照合・往復の移動でE:を4パス使う。D:側のhash、trim、オフラインのファイルシステム検査、
  縮小と起動確認を含め、最初の見積りは6〜10時間とする。再配置に時間がかかる場合や復元する場合は延びる。
  2026-11-22の復旧で約891 GiBのファイルをDiskPartでcompactした実測は、その工程だけで約18分だった。[障害記録](https://github.com/Motoki0705/tennis-lab/issues/2361)

## バックアップ、trim、compact

1. 実行中のjobの終了を待ち、queue、runner、定期起動を一時停止する。Docker、WSLターミナルなど
   distroを再起動し得るアプリを閉じ、確認が済むまで自動起動を止めておく。
   PowerShellから全distroを停止し、一覧で停止を確認する。以後、コマンドの失敗や登録pathの不一致があれば作業を中断する。

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --list --verbose
   ```

2. WindowsのE:にアクセスを制限した新しいディレクトリを作り、停止中のVHDXをコピーする。
   復元に使うSHA-256と元のACLをUbuntuの外に控える。

   ```powershell
   $ErrorActionPreference = 'Stop'
   $original = 'D:\WSL\Ubuntu\ext4.vhdx'
   $backupDir = 'E:\WSL-maintenance\Ubuntu-before'
   $backup = Join-Path $backupDir 'ext4.vhdx'
   if (Test-Path -LiteralPath $backupDir) { throw 'Choose a new backup directory' }
   New-Item -ItemType Directory -Path $backupDir
   Copy-Item -LiteralPath $original -Destination $backup
   $originalHash = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash
   if ((Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash -ne $originalHash) {
       throw 'VHDX backup hash mismatch'
   }
   $originalHash
   (Get-Acl -LiteralPath $original).Sddl
   ```

3. 停止中のUbuntuを、バックアップと同じドライブの新しい作業ディレクトリへ移す。

   ```powershell
   wsl.exe --manage Ubuntu --move 'E:\WSL-maintenance\Ubuntu-work'
   ```

4. 登録の `BasePath` とVHDXがE:の作業先になり、元のVHDXのpathが空いていることを確認する。
   両ドライブで予定どおり空きが確保できたことを測る。
   保守のために `wsl.exe -d Ubuntu` で起動する。WSLはcontainerと判定されるため、
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

2. 作業中のUbuntuを一度起動し、上記Linuxコマンドで確認する。たとえば800 GiBなら `lsblk` に
   `858993459200` bytesと出る。ルートに書込み可能で、guestの空きが十分であることも確認する。
   WSLを終了し、さらに物理容量を回収する必要があれば停止中に `--compact` を再実行する。
   PowerShellで最終的なファイル長とD:の空きを測る。

   ```powershell
   Get-Item -LiteralPath 'E:\WSL-maintenance\Ubuntu-work\ext4.vhdx' | Select-Object Length
   Get-Volume -DriveLetter D | Select-Object Size, SizeRemaining
   ```

3. D:に戻せる空きがあり、元のVHDXのpathが空いていれば移動する。登録pathを確認してUbuntuを起動し、
   ルートのブロックデバイスサイズ、物理D:の空き、E:のミラーmountを確認する。
   既知の `data/` 入力と `ckpt/` を通常のpathから読み込めることと、通常のworktreeの `outputs` が
   E:を指すことも確認する。確認が済むまでバックアップを保持し、その後で定期実行を再開する。

   ```powershell
   wsl.exe --manage Ubuntu --move 'D:\WSL\Ubuntu'
   ```

## 中断・失敗から戻す

1. PowerShellからWSLを停止し、登録の `BasePath` と両方のVHDXのpathを確認してから次の操作を行う。
   Ubuntuの登録は維持する。ディスク変更前の移動に失敗し、登録先のイメージが元のSHA-256と一致するなら、そのイメージを使う。
2. それ以外は、登録中のディスクを変更する前に、バックアップを記録したSHA-256と照合する。
   失敗したイメージや部分コピーをWindowsのE:へ退避して元のファイル名を空ける。
   D:からコピーして空きを作る場合は、退避先のファイル長とSHA-256を照合してからD:側の失敗ファイルを取り除く。
   容量と空の宛先を確認し、検証済みバックアップを `Copy-Item -LiteralPath` で現在の登録先にコピーする。
   `Get-FileHash -Algorithm SHA256` で記録した値と照合し、登録ユーザーとWSLがイメージを使えるownerとACLを確認する。
   登録pathやバックアップの同一性が不明な場合は、jobを停止したまま確認する。
3. E:で復元した場合はバックアップのファイル長でD:の空きを確認し、同じ手順でD:へ戻す。
   空きが足りなければE:に復元した状態で通常実行を止めておく。
   元の仮想容量、mount、入力pathを確認してから再開する。
