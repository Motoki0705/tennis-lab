# Ubuntu VHDX 维护

[English](wsl-vhdx.md) · [日本語](wsl-vhdx.ja.md) · [存储分层](README.zh-CN.md)

由 Windows 主机操作者安排维护时间。保留 `Ubuntu` 的 WSL 2 注册，完成后将磁盘移回
`D:\WSL\Ubuntu\ext4.vhdx`。用拥有该注册的 Windows 账户打开 PowerShell。
工作 VHDX 放在 Windows `E:\`，离线备份放在独立的物理驱动器上。
storage_tier 镜像挂载在 `/mnt/wsl/tennis-data`。

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
- 如果在 D: 就地缩小，需要为 ext4 块重定位期间动态 VHDX 增长到旧上限预留空间。
  WSL 在 ext4 缩小后才降低 VHDX 上限。按实测文件长度 563.78 GiB 计算，需要额外
  `1024 − 563.78 + 20 = 480.22 GiB` 的可用空间，再加分配开销和其他写入。
  D: 实测只有 366.23 GiB 可用，算入其他开销前还差约 114 GiB。trim 和 compact 可以将镜像
  占用的空间释放给 D:，但镜像已占空间与可用空间之和仍小于 1 TiB 加预留空间。按此实测值使用 E: 作为工作盘。
- 设原 VHDX 文件长度为 `B`。在 Windows E: 预留旧虚拟磁盘大小加 `B`，再加分配开销，
  以容纳块重定位期间的增长和保留故障镜像时的原尺寸恢复副本。在另一块物理驱动器上预留 `B` 和开销
  用于备份。按实测值，E: 小计约 1.55 TiB，备份盘约需 564 GiB。C: 有独立的物理盘，
  但实测只有约 77 GiB 可用。接入容量足够的驱动器后再继续。示例使用 `F:`。
- D: 同一时间只需容纳一个镜像。预留 watchdog 阈值和其他写入所需的空间。将 Ubuntu 移到 E: 后，
  确认 D: 仍可容纳原备份及预留空间。移回缩小后的镜像前，D: 可用空间至少要等于其
  **在 Windows 上实测的文件长度**加阈值、其他写入和分配开销。例如，550 GiB 的文件至少需要 570 GiB，
  再加其他写入和分配开销。
- 假设实际速度为 80–150 MiB/s，顺序处理 550–565 GiB 一遍约需 1–2 小时。
  在备份盘上预留写入和哈希验证两遍，在 E: 上预留往返移动两遍。再加上 D: 端哈希、trim、
  离线文件系统检查、缩小和验证，初次安排 6–10 小时。块重定位较慢或需要恢复时会延长。
  2026-11-22 故障恢复时，用 DiskPart 对约 891 GiB 的文件单独进行 compact，实测约 18 分钟。[故障记录](https://github.com/Motoki0705/tennis-lab/issues/2361)

## 跨 Ubuntu 重启暂停自动处理

暂停提交新任务，等待运行中的 CI、训练、dispatcher 和 agent 任务结束。在 Windows 任务计划程序中记录
`TennisLab-WSL-GitHub-Runner` 的启用和运行状态，若已安装则禁用并结束它。关闭 Docker Desktop，
并暂停其他可能打开 Ubuntu 的 Windows 启动程序。
[Windows keepalive 注册脚本](../../../scripts/github_actions/register_wsl_keepalive.ps1)

以运行 user unit 的 Linux 用户执行以下命令，并在 Ubuntu 外保存输出。记录准确的 unit 名称、启用状态、
运行状态和 unit 文件路径。`LoadState=not-found` 表示尚未安装，例如尚未部署的 watchdog。
先解决其他加载错误或 manager 连接错误，再继续。

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

runner 的 unit 名称来自安装脚本读取的文件。watchdog 是 #2372 中的 user service。
storage-tier timer 设置了 `Persistent=true`，启动 timer 时可能补执行错过的周期，因此同时暂停 timer 和 service。
验证所用的 E: 挂载和 CI 只读挂载由 `tennis-data-mount.service` 和 `tennis-lab-actions-assets.service` 提供。
[CI 安装脚本](../../../scripts/github_actions/install_self_hosted_runner.sh) · [运维目标定义](../ops_control/targets.py) · [board unit](../board/systemd/tennis-lab-board-web.service) · [watchdog unit](https://github.com/Motoki0705/tennis-lab/blob/7fce1cbce63a262d3c1746a8a3d1b72917e18698/.agents/ops/d_capacity/systemd/tennis-lab-d-capacity.service) · [storage-tier timer](systemd/tennis-lab-storage-tier.timer) · [timer 行为](https://www.freedesktop.org/software/systemd/man/255/systemd.timer.html#Persistent=)

对于本地 queue，从主 checkout 使用原 worker 的 `TRAINING_QUEUE_DIR` 保存 `status` 输出。
如果通过临时 user service 运行，从已加载的 unit 中按 `ExecStart` 中的 `training_queue.sh serve` 确认实际名称。
记录原启动方式和参数，再对同一 queue 目录运行 `stop`，确认运行中的任务和 worker 都已结束。
同时结束或暂停现有的临时 worker 监控和其他定时启动程序。
[本地 queue 操作](../../skills/training-queue/SKILL.md#subcommands) · [worker 监控 unit 命名](../parent/worker_monitor.py)

在下列各目录中新建并记录 `zz-tennis-lab-vhdx-maintenance.conf`，需要时先创建目录。
使用运行 user unit 的用户的 `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user`。
如果已存在同名文件，先检查上次维护的状态。

- 每个已记录且已安装的 system unit 的 `/etc/systemd/system/<unit>.d/`。
- 用户配置目录中的 `tennis-lab-.service.d/` 和 `tennis-lab-.timer.d/`。
  systemd 将这些带短横线的前缀 drop-in 应用于匹配的 user unit。

```ini
[Unit]
ConditionPathExists=!/
```

由于 `/` 存在，条件会跳过启动。文件保存在 Ubuntu 文件系统中，WSL 关闭后仍然有效，并保留原启用状态和安装链接。
在记录状态时使用的同一 shell 中，让两个 manager 重载配置并停止 timer。检查每条命令的退出状态。
[systemd drop-in 与条件](https://www.freedesktop.org/software/systemd/man/255/systemd.unit.html) · [安装链接的行为](../ops_control/README.md#操作)

```bash
sudo systemctl daemon-reload
systemctl --user daemon-reload
systemctl --user stop 'tennis-lab-*.timer'
```

等待运行中的 storage-tier 等一次性任务结束，再停止已无任务的 service。

```bash
sudo systemctl stop "$runner_service"
sudo systemctl stop tennis-lab-training-queue.service
systemctl --user stop 'tennis-lab-*.service'
```

用 `systemctl show --property=Id,LoadState,ActiveState,DropInPaths <unit>` 检查记录中的 system unit，
检查 user unit 时加 `--user`。确认每个已安装且可启动的 unit 都加载了维护 drop-in 并处于 inactive。
用相同 scope 的 `systemctl cat <unit>` 核对已加载配置中的条件。
同时确认临时 queue、worker 监控已停止，Windows keepalive 已禁用。维护期间**每次**启动 Ubuntu 后立即复查。
遇到意外启动时先处理，再执行 trim 或验证。

## 备份、trim 和 compact

1. 确认上述暂停状态后关闭 WSL 终端。在 Windows PowerShell 中停止所有 distro 并确认均已停止。
   此后若命令失败或注册路径不一致，暂停操作并检查。

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --list --verbose
   ```

2. 核对 Windows 驱动器号及其物理磁盘编号。选择可容纳整个镜像的独立备份盘，核对磁盘编号和实际硬件，
   将下列命令中的 `F` 换成实际驱动器号。在移动或修改原文件之前，完成离线备份并校验 SHA-256。
   在 Ubuntu 外记录备份路径、哈希、原 ACL 和 Linux 用户名。移回 D: 并完成验证前保留此备份。

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

   限制新目录的访问权限，再在同一 PowerShell 会话中复制并校验。

   ```powershell
   Copy-Item -LiteralPath $original -Destination $backup
   $originalHash = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash
   if ((Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash -ne $originalHash) {
       throw 'VHDX backup hash mismatch'
   }
   $originalHash
   (Get-Acl -LiteralPath $original).Sddl
   ```

3. 独立备份通过校验后，将已停止的 Ubuntu 移到 E: 上访问受限的新工作目录。

   ```powershell
   wsl.exe --manage Ubuntu --move 'E:\WSL-maintenance\Ubuntu-work'
   ```

4. 确认注册的 `BasePath` 和 VHDX 已指向工作镜像，原 VHDX 路径已空出，两个卷的可用空间均符合计划要求。
   用 `wsl.exe -d Ubuntu` 启动维护，并复查自动处理仍然暂停。WSL 被识别为容器，因此带有
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

2. 启动工作 Ubuntu，确认自动处理暂停，再重复前述 Linux 检查。以 800 GiB 为例，`lsblk` 应显示
   `858993459200` 字节。确认根文件系统可写且 guest 空间充足。再次关闭 WSL。
   如需进一步回收物理空间，在 Ubuntu 停止时再执行 `--compact`。
   在 PowerShell 中读取最终文件长度和 D: 可用空间。

   ```powershell
   Get-Item -LiteralPath 'E:\WSL-maintenance\Ubuntu-work\ext4.vhdx' | Select-Object Length
   Get-Volume -DriveLetter D | Select-Object Size, SizeRemaining
   ```

3. 确认 D: 的可用空间符合回迁要求且原 VHDX 路径空出，再移回 Ubuntu。检查注册路径并启动，复查自动处理暂停。
   验证根块设备大小、物理 D: 的预留空间、E: 镜像挂载、常用路径下的 `data/` 输入和 `ckpt/` 可读取，
   以及普通 worktree 的 `outputs` 指向 E:。验证通过前保留独立备份，之后恢复正常运行。

   ```powershell
   wsl.exe --manage Ubuntu --move 'D:\WSL\Ubuntu'
   ```

## 从中断或失败中恢复

1. 在 PowerShell 中关闭 WSL，检查注册的 `BasePath` 以及两处 VHDX 路径，再执行下一步。
   如果移动在磁盘修改前失败，且注册路径下的镜像仍与原 SHA-256 一致，可继续使用该镜像。
2. 核对独立备份和已记录的 SHA-256。如果注册路径可用，将失败镜像或不完整副本转存后空出原文件名。
   若从 D: 复制失败文件以腾出空间，先校验转存的文件长度和 SHA-256，再移除 D: 上的失败文件。
   确认空间充足且目标路径为空后，用 `Copy-Item -LiteralPath` 将备份复制到当前注册的路径。
   校验恢复副本的 SHA-256，并确认文件所有者和 ACL 允许注册账户及 WSL 使用镜像。
   若恢复在 E:，用原文件长度确认 D: 空间后再移回。
3. 如果 E: 不可用且注册指向 E:，确认 D: 可容纳原镜像及预留空间。将独立备份复制到 D: 上的新目录，
   同样校验 SHA-256 和权限，再用未占用的恢复名称注册。保留原注册供检查。
   [WSL 的 import-in-place](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#import-a-distribution-in-place)

   ```powershell
   wsl.exe --import-in-place Ubuntu-recovery 'D:\WSL\Ubuntu-recovery\ext4.vhdx'
   ```

4. 用原 Linux 用户启动恢复的 distro。使用恢复名称时，运行 `wsl.exe -d Ubuntu-recovery --user <original-user>`。
   复查自动处理暂停、原虚拟磁盘大小和 D: 可用空间。恢复并验证 E: 镜像的挂载、输入和 worktree 输出路径后再恢复任务。
   使用恢复名称时，保持 Windows 启动程序暂停，直至它们的 distro 参数指向已验证的注册。恢复后的系统验证完成前保留独立备份。

## 恢复正常运行

完成 D: 上的验证后，包括从备份恢复后的验证，仅删除已经记录的维护 drop-in。
运行 `sudo systemctl daemon-reload` 和 `systemctl --user daemon-reload`，将 `UnitFileState` 与原记录核对。
若关机前使用了 `systemctl enable --runtime <unit>` 等临时设置，在原 scope 中恢复。

对于维护前处于 active 的 unit，system unit 通过带 sudo 的 `systemctl start <unit>` 恢复，
user unit 通过 `systemctl --user start <unit>` 恢复。如果此前运行了 watchdog，先启动它并确认
`/run/tennis-lab-d-capacity/status.json` 有新的采样，再接收 CI 任务。先恢复 CI queue，再启动 runner。
使用记录的启动方式恢复本地 queue 和临时 worker 监控，最后恢复 Windows 任务计划程序的原状态。
将运行状态与维护前的记录核对。设置 `Persistent=true` 的 timer 此时可能补跑错过的任务。
