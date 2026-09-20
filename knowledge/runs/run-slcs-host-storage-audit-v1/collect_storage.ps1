# Read-only Windows storage evidence. No elevation, repair, or settings changes.
# Writes one NEW UTF-8 JSON file. Access failures remain explicit in that file.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputPath,

    [ValidateRange(1, 30)]
    [int]$Days = 7,

    [switch]$IncludeReliability
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Get-ErrorInfo {
    param([System.Management.Automation.ErrorRecord]$Record)
    $kind = 'QueryFailed'
    if ($Record.FullyQualifiedErrorId -like '*NoMatchingEventsFound*') {
        $kind = 'NoMatchingEvents'
    }
    elseif ($Record.Exception -is [System.UnauthorizedAccessException] -or
        $Record.CategoryInfo.Category -in @('PermissionDenied', 'SecurityError') -or
        $Record.Exception.HResult -in @(-2147024891, -2147217405) -or
        ($Record.Exception.PSObject.Properties['StatusCode'] -and
            [string]$Record.Exception.StatusCode -eq 'AccessDenied') -or
        $Record.FullyQualifiedErrorId -match 'AccessDenied|UnauthorizedAccess') {
        $kind = 'AccessDenied'
    }
    return [ordered]@{
        kind = $kind
        errorId = $Record.FullyQualifiedErrorId
        exceptionType = $Record.Exception.GetType().FullName
        hresult = $Record.Exception.HResult
        message = $Record.Exception.Message
    }
}

function Invoke-Collection {
    param([scriptblock]$Query)
    try {
        return [ordered]@{ status = 'ok'; data = @(& $Query); error = $null }
    }
    catch {
        $detail = Get-ErrorInfo -Record $_
        return [ordered]@{ status = $detail.kind; data = @(); error = $detail }
    }
}

# CreateNew is atomic: an existing file is never truncated, including a race.
# The parent directory must exist. No temporary copies of diagnostic data are made.
$absoluteOutput = [System.IO.Path]::GetFullPath($OutputPath)
$stream = [System.IO.File]::Open(
    $absoluteOutput, [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write, [System.IO.FileShare]::None
)
$writer = $null
$started = [DateTime]::UtcNow
$since = $started.AddDays(-$Days)
$report = [ordered]@{
    schemaVersion = 1
    startedUTC = $started.ToString('o')
    sinceUTC = $since.ToString('o')
    days = $Days
    reliabilityRequested = [bool]$IncludeReliability
    scope = 'Read-only current-token collection; no automatic elevation.'
    events = [ordered]@{}
    disks = $null
    partitions = $null
    wslDistributions = $null
    scsiPort2 = $null
    reliability = [ordered]@{ status = 'NotRequested'; data = @(); error = $null }
    fatalError = $null
}

try {
    $providers = @(
        'Microsoft-Windows-WHEA-Logger',
        'Microsoft-Windows-Kernel-Power',
        'stornvme', 'disk', 'storahci', 'volmgr',
        'Ntfs', 'Microsoft-Windows-Ntfs',
        'Microsoft-Windows-WER-SystemErrorReporting'
    )
    $eventFields = @(
        'BugcheckCode', 'BugcheckParameter1', 'BugcheckParameter2',
        'BugcheckParameter3', 'BugcheckParameter4', 'SleepInProgress',
        'PowerButtonTimestamp', 'BootAppStatus', 'Checkpoint',
        'ConnectedStandbyInProgress', 'SystemSleepTransitionsToOn',
        'BugcheckInfoFromEFI', 'CheckpointStatus', 'LongPowerButtonPressDetected',
        'DeviceName', 'DeviceNumber', 'DiskNumber', 'VolumeName',
        'Status', 'ErrorCode', 'FailureReason', 'ResetReason', 'Lba',
        'ErrorSource', 'ErrorType', 'ApicId', 'MCABank', 'MciStat',
        'MciAddr', 'MciMisc', 'Bus', 'Device', 'Function', 'VendorID', 'DeviceID'
    )
    foreach ($provider in $providers) {
        $report.events[$provider] = Invoke-Collection {
            $filter = @{ LogName = 'System'; ProviderName = $provider; StartTime = $since }
            if ($provider -eq 'Microsoft-Windows-Kernel-Power') { $filter.Id = 41 }
            # Latest first, at most 100 records per provider, not a total-count query.
            foreach ($event in @(Get-WinEvent -FilterHashtable $filter -MaxEvents 100 -ErrorAction Stop)) {
                $xml = [xml]$event.ToXml()
                $eventData = [ordered]@{}
                foreach ($item in $xml.SelectNodes("/*[local-name()='Event']/*[local-name()='EventData']/*[local-name()='Data']")) {
                    $name = $item.GetAttribute('Name')
                    if ($name -in $eventFields -or
                        ($provider -eq 'Microsoft-Windows-WER-SystemErrorReporting' -and $name -eq 'param1')) {
                        $eventData[$name] = $item.InnerText
                    }
                    elseif (-not $name -and $item.InnerText -match '^\\Device\\') {
                        $eventData['UnnamedDevice'] = $item.InnerText
                    }
                }
                [ordered]@{
                    timeUTC = $event.TimeCreated.ToUniversalTime().ToString('o')
                    provider = $event.ProviderName
                    id = $event.Id
                    level = $event.LevelDisplayName
                    recordId = $event.RecordId
                    message = $event.Message
                    eventData = $eventData
                }
            }
        }
    }

    $report.disks = Invoke-Collection {
        Get-CimInstance -ClassName Win32_DiskDrive `
            -Property Index, Model, FirmwareRevision, SCSIPort, Size, Status -ErrorAction Stop |
            Select-Object Index, Model, FirmwareRevision, SCSIPort, Size, Status
    }
    $report.partitions = Invoke-Collection {
        foreach ($partition in @(Get-CimInstance -ClassName Win32_DiskPartition -ErrorAction Stop)) {
            $mapping = Invoke-Collection {
                Get-CimAssociatedInstance -InputObject $partition `
                    -Association Win32_LogicalDiskToPartition -ErrorAction Stop |
                    Select-Object DeviceID, FileSystem, Size
            }
            [ordered]@{
                diskIndex = $partition.DiskIndex
                partitionIndex = $partition.Index
                deviceID = $partition.DeviceID
                size = $partition.Size
                startingOffset = $partition.StartingOffset
                logicalDrives = $mapping
            }
        }
    }

    # Only the caller's WSL registrations; no account enumeration or environment dump.
    $report.wslDistributions = Invoke-Collection {
        $base = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
        foreach ($key in @(Get-ChildItem -LiteralPath $base -ErrorAction Stop)) {
            $properties = Get-ItemProperty -LiteralPath $key.PSPath -ErrorAction Stop
            $vhd = $null
            if ($properties.PSObject.Properties['VhdFileName']) { $vhd = $properties.VhdFileName }
            [ordered]@{
                distributionName = $properties.DistributionName
                basePath = $properties.BasePath
                vhdFileName = $vhd
            }
        }
    }
    $report.scsiPort2 = Invoke-Collection {
        $base = 'HKLM:\HARDWARE\DEVICEMAP\Scsi\Scsi Port 2'
        $keys = @((Get-Item -LiteralPath $base -ErrorAction Stop))
        $keys += @(Get-ChildItem -LiteralPath $base -Recurse -ErrorAction Stop)
        foreach ($key in $keys) {
            $properties = Get-ItemProperty -LiteralPath $key.PSPath -ErrorAction Stop
            $driver = $null
            $identifier = $null
            if ($properties.PSObject.Properties['Driver']) { $driver = $properties.Driver }
            if ($properties.PSObject.Properties['Identifier']) { $identifier = $properties.Identifier }
            if ($null -ne $driver -or $null -ne $identifier) {
                [ordered]@{ registryKey = $key.Name; driver = $driver; identifier = $identifier }
            }
        }
    }

    if ($IncludeReliability) {
        $report.reliability = Invoke-Collection {
            foreach ($disk in @(Get-PhysicalDisk -ErrorAction Stop)) {
                $counters = Invoke-Collection {
                    $disk | Get-StorageReliabilityCounter -ErrorAction Stop |
                        Select-Object DeviceId, Temperature, TemperatureMax, Wear,
                            PowerOnHours, ReadErrorsTotal, ReadErrorsCorrected,
                            ReadErrorsUncorrected, WriteErrorsTotal,
                            WriteErrorsCorrected, WriteErrorsUncorrected,
                            ReadLatencyMax, WriteLatencyMax, FlushLatencyMax
                }
                [ordered]@{
                    deviceId = $disk.DeviceId
                    friendlyName = $disk.FriendlyName
                    mediaType = [string]$disk.MediaType
                    busType = [string]$disk.BusType
                    healthStatus = [string]$disk.HealthStatus
                    operationalStatus = @($disk.OperationalStatus | ForEach-Object { [string]$_ })
                    counters = $counters
                }
            }
        }
    }
}
catch {
    $report.fatalError = Get-ErrorInfo -Record $_
}
finally {
    $report['finishedUTC'] = [DateTime]::UtcNow.ToString('o')
    try {
        $writer = New-Object System.IO.StreamWriter($stream, (New-Object System.Text.UTF8Encoding($false)))
        $writer.WriteLine(($report | ConvertTo-Json -Depth 14))
        $writer.Flush()
        $stream.Flush($true)
    }
    finally {
        if ($null -ne $writer) { $writer.Dispose() }
        else { $stream.Dispose() }
    }
}
