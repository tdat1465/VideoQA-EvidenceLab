# Download the single self-contained diagnostic report, after worker exit.
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9]+$')][string]$JobId,
    [string]$SshTarget = 'lnthanh03@172.29.74.81',
    [string]$Destination = 'E:\Codex\Documents\ChatGPT\KLTN\outputs\ridge-event-drift-results-v1'
)
$ErrorActionPreference = 'Stop'
if ($SshTarget -notmatch '^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+$') { throw 'Invalid SSH target.' }
[void](New-Item -ItemType Directory -Path $Destination -Force)
$localRoot = (Resolve-Path -LiteralPath $Destination).Path
$temporary = Join-Path $localRoot "drift-$JobId.partial.json"
$target = Join-Path $localRoot "drift-$JobId.json"
$remote = '/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-event-drift-o2f-v1/session-' + $JobId + '/report.json'
& scp -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 ($SshTarget + ':' + $remote) $temporary
if ($LASTEXITCODE -ne 0) { throw 'Report download failed. Existing reports are retained.' }
$report = Get-Content -LiteralPath $temporary -Encoding UTF8 -Raw | ConvertFrom-Json
if ($report.schema -ne 'ridge-event-drift-diagnostic-v1' -or $report.target -ne 'longvideobench:val:o2F-N42Ufo4_2' -or $report.evaluation_results -ne $false) {
    throw 'Unexpected diagnostic report; existing report is retained.'
}
Move-Item -LiteralPath $temporary -Destination $target -Force
Write-Host "Diagnostic complete=$($report.complete); full-run snapshot unchanged=$($report.snapshot_unchanged)"
Write-Host "Da tai vao: $target"
