param(
    [string]$Distribution = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$TaskName = "TennisLab-WSL-GitHub-Runner"
$CurrentIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$WslExecutable = Join-Path $env:WINDIR "System32\wsl.exe"
$WslArguments = "-d $Distribution --exec /opt/tennis-lab-actions/bin/wsl_keepalive.sh"

$Action = New-ScheduledTaskAction -Execute $WslExecutable -Argument $WslArguments
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentIdentity
$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentIdentity `
    -LogonType Interactive `
    -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Description "Keep Ubuntu WSL active for the tennis-lab GitHub runner." `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Registered and started Windows task: $TaskName"
