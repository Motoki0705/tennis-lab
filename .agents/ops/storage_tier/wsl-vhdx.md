# Ubuntu VHDX maintenance

[日本語](wsl-vhdx.ja.md) · [简体中文](wsl-vhdx.zh-CN.md) · [Storage tiers](README.md)

Schedule this procedure with the person operating the Windows host. It keeps the
`Ubuntu` WSL 2 registration and returns its disk to `D:\WSL\Ubuntu\ext4.vhdx`.
Use Windows PowerShell under the Windows account that owns the registration.
Use Windows `E:\` for the working VHDX and an independent physical drive for
the offline backup. The storage-tier mirror is mounted at `/mnt/wsl/tennis-data`.

## Choose the limit and reserve space

In PowerShell, confirm the WSL version, available commands, registered disk
path and host capacity. This procedure uses WSL 3.0.5. If the version has
changed, check that its resize still supports shrinking. Repeat the
registration check after each move.

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

In Ubuntu, confirm that `/` is ext4 on a whole disk and record the virtual disk
size (currently 1 TiB) and guest and host free space in bytes.

```bash
findmnt -nro SOURCE,FSTYPE,OPTIONS --mountpoint /
lsblk -bdno TYPE,SIZE "$(findmnt -nro SOURCE --mountpoint /)"
df -B1 / /mnt/d
```

- Choose a limit above the used ext4 space with room for ext4 overhead,
  reserved blocks and future guest writes. Keep the limit strictly below the
  measured size of physical D: (931.5 GiB here). Subtract other D: usage,
  VHDX metadata and the watchdog reserve (20 GiB by default) when budgeting
  for D: at maximum growth. For example, `800GB` in the WSL CLI means 800 GiB
  and leaves 131.5 GiB before those deductions. Recalculate at maintenance time.
- For an in-place resize on D:, budget for the dynamic VHDX to grow up to its
  old limit during ext4 block relocation. WSL reduces that limit only after
  the ext4 shrink. With a file length of 563.78 GiB, the conservative allowance
  is `1024 − 563.78 + 20 = 480.22 GiB` of additional free space, plus allocation
  overhead and other writes. D: had 366.23 GiB free, about 114 GiB short before
  overhead. Trim and compact can convert occupied image bytes into D: free
  bytes, but the combined budget is still below 1 TiB plus reserve. Use E:
  for the working image with these measurements.
- Let `B` be the original VHDX file length. On E:, reserve the old virtual
  size plus `B` and allocation overhead. This allows for relocation and an
  original-size restore alongside a failed image. Reserve another `B` plus
  overhead on an independent physical backup drive. With the measured values,
  the subtotals are about 1.55 TiB on E: and 564 GiB on the backup drive.
  C: has its own physical disk and had only about 77 GiB free. Attach a drive
  with enough space before proceeding. The backup drive is `F:` in the examples.
- D: needs room for only one image at a time. Keep the watchdog reserve plus room
  for other writes. After moving Ubuntu to E:, confirm D: can still accept the
  original backup with this reserve. Before returning the shrunk VHDX, require
  D: free space of at least its **measured Windows file length** plus that
  reserve and room for other writes and allocation overhead. For a 550 GiB file,
  that is at least 570 GiB before the additional allowance.
- At an assumed 80–150 MiB/s, one 550–565 GiB pass takes about 1–2 hours.
  Budget two passes on the backup drive for copy and hash verification, two
  transfers through E: for both moves, plus the D: hash, trim, offline check,
  resize and validation. Start with a 6–10 hour window and allow more time if
  data relocation or a restore takes longer. On 2026-11-22, the DiskPart compact alone took about 18 minutes
  for a roughly 891 GiB file. See the [incident record](https://github.com/Motoki0705/tennis-lab/issues/2361).

## Pause automation across Ubuntu restarts

Pause new submissions and let active CI, training, dispatcher and agent jobs
finish. In Windows Task Scheduler, record the enabled and running states of
`TennisLab-WSL-GitHub-Runner`, then disable and end it if present. Close Docker
Desktop and pause any other Windows launchers that can open Ubuntu.
[Windows keepalive installer](../../../scripts/github_actions/register_wsl_keepalive.ps1)

As the Linux account that runs the user units, save the following output outside
Ubuntu. It records the exact names, enablement, running state and unit file
locations. A unit with `LoadState=not-found`, such as the watchdog before
deployment, is absent. Resolve other load errors or manager connection errors
before proceeding.

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

The runner name comes from its installer. The watchdog is a user service from
#2372. The user list includes both the storage-tier timer and its service because
`Persistent=true` can run a missed cycle as soon as the timer starts. Keep the
E: mount and CI asset mounts available for verification through
`tennis-data-mount.service` and `tennis-lab-actions-assets.service`.
[CI installer](../../../scripts/github_actions/install_self_hosted_runner.sh) · [Ops targets](../ops_control/targets.py) · [Board unit](../board/systemd/tennis-lab-board-web.service) · [Watchdog unit](https://github.com/Motoki0705/tennis-lab/blob/7fce1cbce63a262d3c1746a8a3d1b72917e18698/.agents/ops/d_capacity/systemd/tennis-lab-d-capacity.service) · [Storage-tier timer](systemd/tennis-lab-storage-tier.timer) · [Timer behavior](https://www.freedesktop.org/software/systemd/man/255/systemd.timer.html#Persistent=)

For the local queue, record its `status` from the main checkout using the
worker's existing `TRAINING_QUEUE_DIR`. If it runs under a transient user service,
identify the exact loaded unit by its `ExecStart` for `training_queue.sh serve`.
Record the original launcher and arguments, then use `stop` with the same queue
directory and confirm its jobs and worker have exited. Finish or pause transient
worker monitors and any other scheduled launchers in use.
[Local queue controls](../../skills/training-queue/SKILL.md#subcommands) · [Worker monitor unit naming](../parent/worker_monitor.py)

Create and record a new file named `zz-tennis-lab-vhdx-maintenance.conf` in each
directory below, creating the directories as needed. Use the owner's user
configuration directory (`${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user`).
Resolve an existing file of the same name before proceeding.

- `/etc/systemd/system/<unit>.d/` for each installed system unit recorded above.
- `tennis-lab-.service.d/` and `tennis-lab-.timer.d/` under the user configuration
  directory. Systemd applies these dash-prefix drop-ins to matching user units.

```ini
[Unit]
ConditionPathExists=!/
```

The condition skips activation because `/` exists. These files persist on the
Ubuntu filesystem across WSL shutdowns and preserve unit enablement and existing
installer links. In the same shell as the inventory, reload both managers and
stop the timers. Check the exit status of each command.
[Systemd drop-ins and conditions](https://www.freedesktop.org/software/systemd/man/255/systemd.unit.html) · [Installer link behavior](../ops_control/README.md#操作)

```bash
sudo systemctl daemon-reload
systemctl --user daemon-reload
systemctl --user stop 'tennis-lab-*.timer'
```

Wait for storage-tier and other running one-shot jobs to finish, then stop the
drained services.

```bash
sudo systemctl stop "$runner_service"
sudo systemctl stop tennis-lab-training-queue.service
systemctl --user stop 'tennis-lab-*.service'
```

Run `systemctl show --property=Id,LoadState,ActiveState,DropInPaths <unit>` for
the recorded system units and the same command with `--user` for the user units.
Check that each installed, launchable unit loads the maintenance drop-in and
is inactive. Use `systemctl cat <unit>` in the same scope to check the condition
in the loaded configuration. Also check the transient queue and worker monitor processes are
stopped and the Windows keepalive is disabled. Repeat these checks immediately
after **every** Ubuntu boot during maintenance. Resolve an unexpected start
before any trim or validation.

## Back up, trim and compact

1. After confirming the durable pauses above, close WSL terminals. From Windows
   PowerShell, stop all distros and confirm they are stopped. At each later
   step, stop if a command fails or the registered path disagrees.

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --list --verbose
   ```

2. Check the Windows drive letters and their physical disk numbers. Use a
   separate device for the backup with space for a full copy and a private
   destination. Confirm the device using its disk number and backing hardware.
   Substitute its actual drive letter for `F` in the commands below.
   Complete and SHA-256 verify the cold backup before moving or modifying the
   original. Record the backup path, hash, original ACL and Linux username outside
   Ubuntu. Keep this backup through verification after returning to D:.

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

   Restrict access to the new directory, then copy and verify in the same
   PowerShell session.

   ```powershell
   Copy-Item -LiteralPath $original -Destination $backup
   $originalHash = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash
   if ((Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash -ne $originalHash) {
       throw 'VHDX backup hash mismatch'
   }
   $originalHash
   (Get-Acl -LiteralPath $original).Sddl
   ```

3. With the independently verified backup in place, move the stopped distro
   to a fresh, access-controlled directory on E:.

   ```powershell
   wsl.exe --manage Ubuntu --move 'E:\WSL-maintenance\Ubuntu-work'
   ```

4. Confirm the registered `BasePath` and VHDX now point to the working image,
   the original VHDX path is vacant, and both volumes have the planned free
   space. Start `wsl.exe -d Ubuntu` for maintenance and verify the automation
   pauses again before trimming. Ubuntu's `fstrim.timer` is skipped because WSL is
   detected as a container and the timer has `ConditionVirtualization=!container`.
   Trim the mounted root explicitly, then exit Ubuntu.

   ```bash
   systemd-detect-virt --container
   systemctl cat fstrim.timer
   systemctl show fstrim.timer -p ActiveState -p ConditionResult
   sudo fstrim --verbose /
   exit
   ```

5. Shut down WSL again and compact the working VHDX. Compare its Windows
   `Length` before and after. Compact reclaims physical storage and leaves
   the virtual capacity at its previous limit.

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --manage Ubuntu --compact
   ```

## Shrink ext4 and the VHDX, then return to D:

1. In PowerShell, ensure Ubuntu is stopped. Substitute the planned size for the
   example if needed. In WSL 3.0.5, `--resize` runs `e2fsck` and `resize2fs`
   on the offline ext4 disk in the WSL utility VM. After filesystem shrink
   succeeds, WSL detaches the disk and reduces the VHDX limit. An error in the
   second phase can leave ext4 already smaller. [WSL host implementation](https://github.com/microsoft/WSL/blob/3.0.5/src/windows/service/exe/LxssUserSession.cpp#L1854-L1911) · [Utility VM implementation](https://github.com/microsoft/WSL/blob/3.0.5/src/linux/init/main.cpp#L2940-L2998)

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --manage Ubuntu --resize 800GB
   ```

2. Boot the working Ubuntu, verify the automation pauses, and repeat the Linux checks above. Expect
   `858993459200` bytes from `lsblk` for 800 GiB, a writable root and adequate
   guest free space. Shut down again. If more physical space must be reclaimed,
   repeat `--compact` while Ubuntu is stopped. Read the final file length and D:
   free space in PowerShell.

   ```powershell
   Get-Item -LiteralPath 'E:\WSL-maintenance\Ubuntu-work\ext4.vhdx' | Select-Object Length
   Get-Volume -DriveLetter D | Select-Object Size, SizeRemaining
   ```

3. Once the D: return space check passes and the original VHDX path is vacant,
   move Ubuntu back. Confirm the registered path and boot it again. Verify the automation pauses,
   root block-device size, physical D: reserve, and the E: mirror mount.
   Check that a known `data/` input and `ckpt/` load from their usual paths and
   that a normal worktree's `outputs` resolves to E:. Keep the independent
   backup through these checks, then restore normal operation.

   ```powershell
   wsl.exe --manage Ubuntu --move 'D:\WSL\Ubuntu'
   ```

## Restore after an interruption or failure

1. Stop WSL from PowerShell and check the registered `BasePath` and both disk
   paths before trying another operation.
   If a move failed before any disk changes and the registered image still
   matches the original SHA-256, use that intact copy.
2. Recheck the independent backup against the saved SHA-256. If the registered
   path is available, preserve any failed or partial image before freeing its
   filename. If you copy a failed file off D:, compare length and SHA-256 before
   removing it there. With sufficient space and a vacant destination, copy the
   backup with `Copy-Item -LiteralPath` to the registered path. Verify its SHA-256
   and ensure the file owner and ACL allow the registered account and WSL to use
   it. If it is on E:, move it back using the D: space check for the original
   file length.
3. If E: is unavailable and the registration points there, check D: can fit the
   original image plus reserve. Copy the backup to a fresh directory on D:,
   verify SHA-256 and permissions as above, and register the copy under an unused
   recovery name. Leave the original registration in place for inspection.
   [WSL import-in-place](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#import-a-distribution-in-place)

   ```powershell
   wsl.exe --import-in-place Ubuntu-recovery 'D:\WSL\Ubuntu-recovery\ext4.vhdx'
   ```

4. Boot the restored distro with the original Linux user. For the recovery name,
   use `wsl.exe -d Ubuntu-recovery --user <original-user>`. Verify the automation
   pauses, original virtual size and D: free space. Restore and check the E:
   mirror mount, inputs and worktree output paths before resuming workloads.
   If using the recovery name, keep Windows launchers paused until their distro
   references point to the verified registration.
   Keep the independent backup until the restored system has passed these checks.

## Restore normal operation

After validation on D: (including after a rollback), remove only the maintenance
drop-ins you recorded. Run `sudo systemctl daemon-reload` and
`systemctl --user daemon-reload`. Compare `UnitFileState` against the inventory.
Reapply any original runtime-only settings lost during shutdown, such as
`systemctl enable --runtime <unit>`, in their original scope.

Use `systemctl start <unit>` with sudo for system units and
`systemctl --user start <unit>` for user units that were active before maintenance.
Start the watchdog first if it previously ran, and check for a fresh sample in
`/run/tennis-lab-d-capacity/status.json` before admitting CI jobs. Bring back the
CI queue before the runner. Resume the local queue and transient monitors through
their recorded launchers. Restore the prior Windows Task Scheduler state last.
Check active states against the saved inventory. Timers with `Persistent=true`
can start missed jobs at this point.
