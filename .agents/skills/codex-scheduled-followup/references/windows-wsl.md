# Observed Windows / WSL setup

Read only for host, app-startup or storage troubleshooting. These observations came from OpenAI.Codex 26.930.7945.0 on 2026-10-07, not a public file-format API guarantee.

## Home, interpreter and working directory

The observed WSL CODEX_HOME was /mnt/c/Users/kamim/.codex, shared with Windows C:\Users\kamim\.codex. Verify the app's home on each machine; WSL homes are not necessarily shared. Use explicit --codex-home and an existing interpreter path rather than changing environment variables.

A Windows executor cannot use /home/... or a synthetic C:\home\... as its working directory. If a future monitoring turn runs in Windows but the job is in WSL, retain the verified distro, Linux working directory, interpreter and a real Windows working directory in the handoff. From that Windows directory use wsl.exe -d DISTRO --cd LINUX_WORKTREE -- LINUX_PYTHON ... . Pass arguments directly; nested PowerShell/Bash command strings can expand variables or introduce CRLF errors. The original monitor needed this host correction before it could read its queue.

## Desktop app startup from WSL

The package was OpenAI.Codex, but the GUI executable was **ChatGPT.exe**. codex.exe alone may just be an app server. Inspect the installed package and GUI before launching:

```bash
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  'Get-AppxPackage -Name OpenAI.Codex | Select-Object PackageFamilyName,InstallLocation | ConvertTo-Json -Compress'
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  'Get-Process ChatGPT -ErrorAction SilentlyContinue | Select-Object ProcessName,Id,Path | ConvertTo-Json -Compress'
```

AppxManifest.xml identified application ID App and family OpenAI.Codex_2p2nqsd0c76g0, so this installation was opened with:

```bash
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  'Start-Process explorer.exe -ArgumentList "shell:AppsFolder\OpenAI.Codex_2p2nqsd0c76g0!App"'
```

Check identity before reusing that command elsewhere. Starting the GUI is a dependency, not a workaround for permissions/account restrictions. Do not unpack the app or inspect its binaries for ordinary setup; use the native tool/UI if this fallback stops being accepted.

## Verification evidence and limits

The installed app's bootstrap-C8gUBg5L.js was inspected to establish the version-1 heartbeat file import and next-run behavior. The app imported the original configuration itself; neither SQLite rows nor binaries were modified. Direct WSL access to the live Windows SQLite file produced a disk I/O error, so verification now copies the database/WAL to a stable temporary snapshot.

[Official scheduled-task documentation](https://learn.chatgpt.com/docs/automations?surface=app) covers same-chat continuations and keeping the computer/app running for local work. It does not document this local file-write route. Acceptance by another app version or account migration state is unverified until its own registration check succeeds.
