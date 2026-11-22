# Ubuntu VHDX maintenance

[日本語](wsl-vhdx.ja.md) · [简体中文](wsl-vhdx.zh-CN.md) · [Storage tiers](README.md)

Schedule this procedure with the person operating the Windows host. It keeps the
`Ubuntu` WSL 2 registration and returns its disk to `D:\WSL\Ubuntu\ext4.vhdx`.
Use Windows PowerShell under the Windows account that owns the registration.
Use a private directory on Windows `E:\` for the offline backup and working VHDX.
The storage-tier mirror is a separate ext4 mount at `/mnt/wsl/tennis-data`.

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
- Let `B` be the original VHDX file length. Reserve at least `2 × B + old virtual
  size` on Windows E:, plus filesystem and VHDX overhead. This covers the backup,
  growth during block relocation, and a restored copy while retaining a failed
  image. With an approximately 564 GiB file and a 1 TiB limit, the subtotal is
  about 2.1 TiB.
- D: needs room for only one image at a time. Keep the watchdog reserve plus room
  for other writes. After moving Ubuntu to E:, confirm D: can still accept the
  original backup with this reserve. Before returning the shrunk VHDX, require
  D: free space of at least its **measured Windows file length** plus that
  reserve and room for other writes and allocation overhead. For a 550 GiB file,
  that is at least 570 GiB before the additional allowance.
- At an assumed 80–150 MiB/s, one 550–565 GiB pass through the HDD takes about
  1–2 hours. Budget four E: passes for backup, backup hash and both moves, plus
  the D: hash, trim, offline filesystem check, resize and validation. Start
  with a 6–10 hour window and allow more time if data relocation or a restore
  is needed. On 2026-11-22, the DiskPart compact alone took about 18 minutes
  for a roughly 891 GiB file. See the [incident record](https://github.com/Motoki0705/tennis-lab/issues/2361).

## Back up, trim and compact

1. Drain active jobs, pause queues, runners and scheduled launches, and close
   Docker, WSL terminals and any other application that can reopen a distro.
   Keep automatic launches paused through verification. From PowerShell, stop
   all distros and confirm they are stopped. At each later step, stop the
   procedure if a command fails or the recorded path disagrees.

   ```powershell
   wsl.exe --shutdown
   if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed' }
   wsl.exe --list --verbose
   ```

2. Make a cold copy into a new, access-controlled directory on Windows E:.
   Record the SHA-256 and original ACL outside Ubuntu for rollback.

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

3. Move the stopped distro to a fresh directory beside the backup.

   ```powershell
   wsl.exe --manage Ubuntu --move 'E:\WSL-maintenance\Ubuntu-work'
   ```

4. Confirm the registered `BasePath` and VHDX now point to the working image,
   the original VHDX path is vacant, and both volumes have the planned free
   space. Start `wsl.exe -d Ubuntu` for
   maintenance only. Ubuntu's `fstrim.timer` is skipped because WSL is
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

2. Boot the working Ubuntu once and repeat the Linux checks above. Expect
   `858993459200` bytes from `lsblk` for 800 GiB, a writable root and adequate
   guest free space. Shut down again. If more physical space must be reclaimed,
   repeat `--compact` while Ubuntu is stopped. Read the final file length and D:
   free space in PowerShell.

   ```powershell
   Get-Item -LiteralPath 'E:\WSL-maintenance\Ubuntu-work\ext4.vhdx' | Select-Object Length
   Get-Volume -DriveLetter D | Select-Object Size, SizeRemaining
   ```

3. Once the D: return space check passes and the original VHDX path is vacant,
   move Ubuntu back. Confirm the registered path and boot it again. Verify the
   root block-device size, physical D: reserve, and the E: mirror mount.
   Check that a known `data/` input and `ckpt/` load from their usual paths and
   that a normal worktree's `outputs` resolves to E:. Keep the backup until
   these checks pass, then resume scheduled work.

   ```powershell
   wsl.exe --manage Ubuntu --move 'D:\WSL\Ubuntu'
   ```

## Restore after an interruption or failure

1. Stop WSL from PowerShell and check the registered `BasePath` and both disk
   paths before trying another operation. Keep the same Ubuntu registration.
   If a move failed before any disk changes and the registered image still
   matches the original SHA-256, use that intact copy.
2. Otherwise, verify the backup against the recorded SHA-256 before changing
   the registered disk. Preserve the failed or partial image on Windows E: and
   clear its original filename. If copying it off D: to make room, compare
   length and SHA-256 before removing the failed file from D:. After confirming
   space and an empty destination, copy the verified backup to the currently
   registered VHDX path using `Copy-Item -LiteralPath`. Check the restored copy
   against the recorded SHA-256 with `Get-FileHash -Algorithm SHA256` and ensure
   the file owner and ACL allow the registered account and WSL to use the image.
   Keep workloads paused if the registered path or backup identity is uncertain.
3. If the restored Ubuntu is on E:, move it back to D: using the same space check
   with the backup's file length. If space is insufficient, keep the restore
   on E: and keep the normal workloads paused. Verify the original virtual size,
   mount and input paths before resuming.
