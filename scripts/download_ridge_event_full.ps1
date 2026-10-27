# Continue an interrupted export download. No experiments or server files are changed.
# Windows PowerShell 5.1 and PowerShell 7; requires the existing trusted SSH host key.
param(
    [string]$SshTarget = 'lnthanh03@172.29.74.81',
    [string]$Destination = 'E:\Codex\Documents\ChatGPT\KLTN\outputs\ridge-event-full1337-results-' + [DateTime]::Now.ToString('yyyyMMdd-HHmmss'),
    [string[]]$ReuseFrom = @()
)
$ErrorActionPreference = 'Stop'
$runName = 'lvb-ridge-event-coverage-full1337-v1'
if ($SshTarget -notmatch '^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+$') {
    throw 'SshTarget must have the form user@host.'
}
if (-not (Test-Path -LiteralPath $Destination -PathType Container)) {
    [void](New-Item -ItemType Directory -Path $Destination -Force)
}
$destinationRoot = (Resolve-Path -LiteralPath $Destination).Path
$sshOptions = @('-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3')

function Get-CheckedLocalPath([string]$relativePath, [string]$rootPath) {
    # Remote inventory accepts only known ASCII export paths, never arbitrary paths.
    if ($relativePath -notmatch '^(logs/[A-Za-z0-9_.-]+|lvb-ridge-event-coverage-full1337-v1/((curves|scores)/[a-f0-9]{64}\.json|[A-Za-z0-9_.-]+))$') {
        throw "Unexpected inventory path: $relativePath"
    }
    $resolvedRoot = [IO.Path]::GetFullPath($rootPath).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $resolvedPath = [IO.Path]::GetFullPath((Join-Path $resolvedRoot ($relativePath.Replace('/', [IO.Path]::DirectorySeparatorChar))))
    if (-not $resolvedPath.StartsWith($resolvedRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Inventory path escaped the destination: $relativePath"
    }
    return $resolvedPath
}

function Test-CompleteFile([string]$path, $entry) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $path).Length -ne [long]$entry.bytes) { return $false }
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $entry.sha256
}

Write-Host "Continue into: $destinationRoot"
Write-Host 'Step 1/3: read the server file inventory and SHA256 hashes.'
Write-Host 'At a password prompt, enter the server password; input is hidden.'
Write-Host 'This run uses one SSH inventory connection and, if needed, one SFTP transfer connection.'
$remoteInventoryScript = @'
from pathlib import Path
import hashlib, json
run = Path('/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-event-coverage-full1337-v1')
logs = Path('/media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1/logs')
run_name = 'lvb-ridge-event-coverage-full1337-v1'
required = ['summary.json', 'contract.json', 'manifest.json', 'predictions.jsonl']
missing = [name for name in required if not (run / name).is_file()]
if missing:
    raise SystemExit('Missing required server exports: ' + ', '.join(missing))
files = {}
for name in required + ['environment.latest.txt', 'gpu.latest.txt']:
    path = run / name
    if path.is_file():
        files[run_name + '/' + name] = path
for pattern in ['gpu-util-*.csv', 'gpu-util-*.err', 'gpu-context-*.json', 'phases-*.log']:
    for path in run.glob(pattern):
        if path.is_file():
            files[run_name + '/' + path.name] = path
for kind in ['curves', 'scores']:
    for path in (run / kind).glob('*.json'):
        if path.is_file():
            files[run_name + '/' + kind + '/' + path.name] = path
for pattern in ['ridge-evtfull-*.out', 'ridge-evtfull-*.err', 'ridge-event-full-*.json']:
    for path in logs.glob(pattern):
        if path.is_file():
            files['logs/' + path.name] = path
entries = []
for relative, path in sorted(files.items()):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SystemExit('Export changed during inventory; download after worker exits: ' + str(path))
    entries.append({'relative': relative, 'remote': str(path), 'bytes': before.st_size, 'sha256': digest.hexdigest()})
print('__RIDGE_RESUME_INVENTORY__' + json.dumps({'schema': 1, 'files': entries}, separators=(',', ':')))
'@
$inventoryOutput = @($remoteInventoryScript | & ssh @sshOptions $SshTarget 'python3 -')
if ($LASTEXITCODE -ne 0) { throw 'Cannot read server inventory. Existing downloads are unchanged; rerun this script after reconnecting.' }
$inventoryLines = @($inventoryOutput | Where-Object { $_ -is [string] -and $_.StartsWith('__RIDGE_RESUME_INVENTORY__') })
if ($inventoryLines.Count -ne 1) { throw 'No valid server inventory received. Existing files are retained.' }
$inventory = $inventoryLines[0].Substring('__RIDGE_RESUME_INVENTORY__'.Length) | ConvertFrom-Json
if ($inventory.schema -ne 1 -or @($inventory.files).Count -eq 0) { throw 'Invalid or empty inventory.' }

$seen = @{}
foreach ($entry in $inventory.files) {
    [void](Get-CheckedLocalPath $entry.relative $destinationRoot)
    $expectedRemote = if ($entry.relative.StartsWith('logs/')) {
        '/media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1/' + $entry.relative
    } else {
        '/media/lnthanh03/DatHa/runs/videoqa/' + $entry.relative
    }
    if ($seen.ContainsKey($entry.relative) -or $entry.remote -cne $expectedRemote -or
        $entry.sha256 -notmatch '^[a-f0-9]{64}$' -or [long]$entry.bytes -lt 0) {
        throw "Invalid inventory entry: $($entry.relative)"
    }
    $seen[$entry.relative] = $true
}
foreach ($name in @('summary.json', 'contract.json', 'manifest.json', 'predictions.jsonl')) {
    if (-not $seen.ContainsKey("$runName/$name")) { throw "Incomplete inventory: $name" }
}
$inventoryPath = Join-Path $destinationRoot 'resume-download-inventory.json'
$inventory | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $inventoryPath -Encoding UTF8

Write-Host 'Step 2/3: keep complete files and reuse matching files from older downloads.'
$missingFiles = @()
$kept = 0
$reused = 0
foreach ($entry in $inventory.files) {
    $targetPath = Get-CheckedLocalPath $entry.relative $destinationRoot
    if (Test-CompleteFile $targetPath $entry) { $kept++; continue }
    $reusePath = $null
    foreach ($sourceRoot in $ReuseFrom) {
        if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) { continue }
        $candidatePath = Get-CheckedLocalPath $entry.relative ((Resolve-Path -LiteralPath $sourceRoot).Path)
        if (Test-CompleteFile $candidatePath $entry) { $reusePath = $candidatePath; break }
    }
    if ($reusePath) {
        [void](New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force)
        Copy-Item -LiteralPath $reusePath -Destination $targetPath -Force
        if (-not (Test-CompleteFile $targetPath $entry)) { throw "Local copy failed verification: $($entry.relative)" }
        $reused++
    } else {
        $missingFiles += $entry
    }
}
Write-Host "Verified existing: $kept; reused locally: $reused; missing or changed: $($missingFiles.Count)"

Write-Host 'Step 3/3: download only missing or changed files, then verify checksums.'
# scp reconnects for each remote source even in a multi-source invocation.
# Feed commands to a single interactive-mode SFTP process instead. Do not use
# -b, which defaults to noninteractive authentication. Passwords stay in the
# native SSH console prompt; no password is stored or sent through stdin.
$downloaded = 0
if ($missingFiles.Count -gt 0) {
    $commands = @()
    foreach ($entry in $missingFiles) {
        $targetPath = Get-CheckedLocalPath $entry.relative $destinationRoot
        [void](New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force)
        $sftpLocalPath = $targetPath.Replace('\', '/')
        if ($sftpLocalPath -match '[\r\n"\x00]') { throw 'Invalid local path for SFTP command.' }
        $commands += 'get -p "' + $entry.remote + '" "' + $sftpLocalPath + '"'
    }
    $commands += 'bye'
    Write-Host "Requesting $($missingFiles.Count) files in ONE SFTP connection."
    $previousOutputEncoding = $OutputEncoding
    try {
        # Native stdin piping in Windows PowerShell 5.1 otherwise uses ASCII.
        $OutputEncoding = New-Object System.Text.UTF8Encoding($false)
        $commands | & sftp @sshOptions -o BatchMode=no $SshTarget
        $transferExit = $LASTEXITCODE
    } finally {
        $OutputEncoding = $previousOutputEncoding
    }
    if ($transferExit -ne 0) {
        throw 'Download interrupted. Complete files are retained; rerun the exact same command to continue.'
    }
    foreach ($entry in $missingFiles) {
        if (-not (Test-CompleteFile (Get-CheckedLocalPath $entry.relative $destinationRoot) $entry)) {
            throw "Missing file or checksum mismatch: $($entry.relative). Existing files are retained; rerun after worker exits."
        }
        $downloaded++
    }
}
foreach ($entry in $inventory.files) {
    if (-not (Test-CompleteFile (Get-CheckedLocalPath $entry.relative $destinationRoot) $entry)) {
        throw "Final check failed: $($entry.relative)"
    }
}
$runPath = Join-Path $destinationRoot $runName
$summary = Get-Content -LiteralPath (Join-Path $runPath 'summary.json') -Raw | ConvertFrom-Json
$predictionCount = @(Get-Content -LiteralPath (Join-Path $runPath 'predictions.jsonl') | Where-Object { $_.Trim() }).Count
if ($predictionCount -ne $summary.completed_variant_answers) { throw 'Prediction count does not match summary; download again after worker exits.' }
Write-Host "Verified all $(@($inventory.files).Count) server files. New downloads: $downloaded."
Write-Host "Variant answers: $predictionCount/6685; complete=$($summary.complete)"
Write-Host "Full questions: $($summary.completed_questions)/1337; complete=$($summary.complete)"
Write-Host "Da tai vao: $destinationRoot"
