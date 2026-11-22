# Ubuntu VHDX 维护

[English](wsl-vhdx.md) · [日本語](wsl-vhdx.ja.md) · [存储分层](README.zh-CN.md)

由 Windows 主机操作者安排维护时间。保留 `Ubuntu` 的 WSL 2 注册，完成后将磁盘移回
`D:\WSL\Ubuntu\ext4.vhdx`。用拥有该注册的 Windows 账户打开 PowerShell。
在 Windows `E:\` 的访问受限目录中存放离线备份和工作 VHDX。
storage_tier 镜像位于另一个 ext4 挂载点 `/mnt/wsl/tennis-data`。

## 选择上限并预留空间

在 PowerShell 中确认 WSL 版本、可用命令、注册路径和主机容量。本流程使用 WSL 3.0.5。
版本变动时确认 resize 仍支持缩小，每次移动后重新检查注册路径。

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

在 Ubuntu 中确认 `/` 使用整块磁盘上的 ext4，以字节记录虚拟磁盘大小（当前为 1 TiB）、
guest 和主机的可用空间。

```bash
findmnt -nro SOURCE,FSTYPE,OPTIONS --mountpoint /
lsblk -bdno TYPE,SIZE "$(findmnt -nro SOURCE --mountpoint /)"
df -B1 / /mnt/d
```

- 上限需容纳当前 ext4 已用空间、ext4 元数据、保留块和未来的 guest 写入，并且严格小于物理 D: 的实测总容量
  （本机为 931.5 GiB）。按 VHDX 增长到上限来规划，从 D: 总容量扣除其他用量、VHDX 元数据及
  watchdog 阈值（默认 20 GiB）。例如，WSL CLI 的 `800GB` 表示 800 GiB，扣除上述预留前还剩
  131.5 GiB。维护时用实测值重新计算。
- 设原 VHDX 文件长度为 `B`，在 Windows E: 预留至少 `2 × B + 原虚拟磁盘大小`，再加文件系统和
  VHDX 开销。这可容纳备份、块重定位时的增长，以及留存故障镜像时的恢复副本。
  如果文件长度约为 564 GiB、上限为 1 TiB，小计约为 2.1 TiB。
- D: 同一时间只需容纳一个镜像。预留 watchdog 阈值和其他写入所需的空间。将 Ubuntu 移到 E: 后，
  确认 D: 仍可容纳原备份及预留空间。移回缩小后的镜像前，D: 可用空间至少要等于其
  **在 Windows 上实测的文件长度**加阈值、其他写入和分配开销。例如，550 GiB 的文件至少需要 570 GiB，
  再加其他写入和分配开销。
- 若 HDD 实际速度为 80–150 MiB/s，顺序处理 550–565 GiB 一遍约需 1–2 小时。
  备份、备份哈希校验和两次移动合计访问 E: 四遍。再加上 D: 端哈希、trim、离线文件系统检查、缩小和验证，
  初次安排 6–10 小时。块重定位较慢或需要恢复时会延长。
  2026-11-22 故障恢复时，用 DiskPart 对约 891 GiB 的文件单独进行 compact，实测约 18 分钟。[故障记录](https://github.com/Motoki0705/tennis-lab/issues/2361)

## 备份、trim 和 compact

1. 等待运行中的任务完成，暂停队列、runner 和定时启动，关闭 Docker、WSL 终端及其他可能重新打开
   distro 的应用。在验证完成之前保持自动启动暂停。在 PowerShell 中停止所有 distro 并确认均已停止。
   此后若命令失败或注册路径不一致，暂停操作并检查。

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --list --verbose
   ```

2. 在 Windows E: 上创建有访问限制的新目录，将已停止的 VHDX 离线复制到该目录。
   在 Ubuntu 外部记录用于恢复的 SHA-256 和原 ACL。

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

3. 将已停止的 Ubuntu 移到备份所在驱动器上的新工作目录。

   ```powershell
   wsl.exe --manage Ubuntu --move 'E:\WSL-maintenance\Ubuntu-work'
   ```

4. 确认注册的 `BasePath` 和 VHDX 已指向工作镜像，原 VHDX 路径已空出，两个卷的可用空间均符合计划要求。
   用 `wsl.exe -d Ubuntu` 启动维护。WSL 被识别为容器，因此带有
   `ConditionVirtualization=!container` 的 `fstrim.timer` 会跳过。
   对已挂载的根文件系统手动运行 trim，然后退出 Ubuntu。

   ```bash
   systemd-detect-virt --container
   systemctl cat fstrim.timer
   systemctl show fstrim.timer -p ActiveState -p ConditionResult
   sudo fstrim --verbose /
   exit
   ```

5. 再次关闭 WSL 并 compact 工作 VHDX。比较 Windows 上操作前后的 `Length`。
   compact 回收物理存储分配，虚拟磁盘上限在下一步调整。

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --manage Ubuntu --compact
   ```

## 缩小 ext4 和 VHDX 后移回 D:

1. 确认 Ubuntu 已停止，再在 PowerShell 中传入计划的大小。WSL 3.0.5 的 `--resize` 在 WSL 实用程序
   VM 中对离线 ext4 磁盘运行 `e2fsck` 和 `resize2fs`。文件系统缩小成功后，WSL 分离磁盘并降低
   VHDX 上限。第二阶段失败时，ext4 可能已经变小。
   [WSL 主机实现](https://github.com/microsoft/WSL/blob/3.0.5/src/windows/service/exe/LxssUserSession.cpp#L1854-L1911) · [实用程序 VM 实现](https://github.com/microsoft/WSL/blob/3.0.5/src/linux/init/main.cpp#L2940-L2998)

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --manage Ubuntu --resize 800GB
   ```

2. 启动一次工作 Ubuntu，重复前述 Linux 检查。以 800 GiB 为例，`lsblk` 应显示
   `858993459200` 字节。确认根文件系统可写且 guest 空间充足。再次关闭 WSL。
   如需进一步回收物理空间，在 Ubuntu 停止时再执行 `--compact`。
   在 PowerShell 中读取最终文件长度和 D: 可用空间。

   ```powershell
   Get-Item -LiteralPath 'E:\WSL-maintenance\Ubuntu-work\ext4.vhdx' | Select-Object Length
   Get-Volume -DriveLetter D | Select-Object Size, SizeRemaining
   ```

3. 确认 D: 的可用空间符合回迁要求且原 VHDX 路径空出，再移回 Ubuntu。检查注册路径并启动，
   验证根块设备大小、物理 D: 的预留空间、E: 镜像挂载、常用路径下的 `data/` 输入和 `ckpt/` 可读取，
   以及普通 worktree 的 `outputs` 指向 E:。验证通过前保留备份，之后恢复定时任务。

   ```powershell
   wsl.exe --manage Ubuntu --move 'D:\WSL\Ubuntu'
   ```

## 从中断或失败中恢复

1. 在 PowerShell 中关闭 WSL，检查注册的 `BasePath` 以及两处 VHDX 路径，再执行下一步。
   保留原 Ubuntu 注册。如果移动在磁盘修改前失败，且注册路径下的镜像仍与原 SHA-256 一致，可继续使用该镜像。
2. 其他情况下，修改注册磁盘前先核对备份和已记录的 SHA-256。
   把失败镜像或不完整副本转存到 Windows E: 以供检查，并空出原文件名。
   若从 D: 复制失败文件以腾出空间，先校验转存的文件长度和 SHA-256，再移除 D: 上的失败文件。
   确认可用空间和目标路径后，用 `Copy-Item -LiteralPath` 将已校验的备份复制到当前注册的 VHDX 路径。
   用 `Get-FileHash -Algorithm SHA256` 对照记录的值，并检查文件所有者和 ACL 是否允许注册账户及 WSL
   使用镜像。注册路径或备份身份不明确时继续暂停工作负载。
3. 若 Ubuntu 恢复在 E:，用备份文件长度检查 D: 的可用空间，再用相同步骤移回 D:。
   空间不足时将恢复后的镜像暂留 E:，继续暂停正常任务。
   恢复前的虚拟磁盘大小、挂载和输入路径都验证通过后，再恢复任务。
