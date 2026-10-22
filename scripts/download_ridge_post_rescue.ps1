param(
    [string]$SshTarget = 'lnthanh03@172.29.74.81',
    [string]$Destination = 'E:\Codex\Documents\ChatGPT\KLTN\outputs\ridge-postrescue-results-' + [DateTime]::Now.ToString('yyyyMMdd-HHmmss')
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Destination) { throw "Destination already exists: $Destination" }
New-Item -ItemType Directory -Path $Destination | Out-Null
$logs = Join-Path $Destination 'logs'
New-Item -ItemType Directory -Path $logs | Out-Null
$repo = '/media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1'
$sources = @("${SshTarget}:${repo}/logs/ridge-post-*.out", "${SshTarget}:${repo}/logs/ridge-post-*.err", "${SshTarget}:${repo}/logs/ridge-postrescue-*.json")
& scp -p -o StrictHostKeyChecking=yes -o ConnectTimeout=15 @sources $logs
if ($LASTEXITCODE -ne 0) { throw "Incomplete logs; keep files in $logs" }
$remote = '/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-postrescue-v1-dev200'
$local = Join-Path $Destination 'lvb-ridge-postrescue-v1-dev200'
New-Item -ItemType Directory -Path $local | Out-Null
$files = foreach ($name in @('summary.json','contract.json','manifest.json','predictions.jsonl','environment.latest.txt','gpu.latest.txt')) { "${SshTarget}:${remote}/${name}" }
& scp -p -o StrictHostKeyChecking=yes -o ConnectTimeout=15 @files $local
if ($LASTEXITCODE -ne 0) { throw "Incomplete exports; logs retained in $logs" }
& scp -r -p -o StrictHostKeyChecking=yes -o ConnectTimeout=15 "${SshTarget}:${remote}/curves" $local
if ($LASTEXITCODE -ne 0) { throw "Incomplete curves/scores; exports retained in $local" }
# Optional for older sessions. Copy telemetry only after existence is checked remotely.
$telemetry = @(& ssh -o StrictHostKeyChecking=yes -o ConnectTimeout=15 $SshTarget "find '$remote' -maxdepth 1 -type f \( -name 'gpu-util-*.csv' -o -name 'gpu-util-*.err' -o -name 'gpu-context-*.json' -o -name 'phases-*.log' \) -print")
if ($LASTEXITCODE -ne 0) { throw 'Cannot list session telemetry; exports retained' }
if ($telemetry.Count -gt 0) {
    $telemetrySources = foreach ($path in $telemetry) { if ($path.Trim()) { "${SshTarget}:$($path.Trim())" } }
    if ($telemetrySources) {
        & scp -p -o StrictHostKeyChecking=yes -o ConnectTimeout=15 @telemetrySources $local
        if ($LASTEXITCODE -ne 0) { throw 'Incomplete telemetry; predictions and curves retained' }
    }
}
$summary = Get-Content -LiteralPath (Join-Path $local 'summary.json') -Raw | ConvertFrom-Json
$count = @(Get-Content -LiteralPath (Join-Path $local 'predictions.jsonl') | Where-Object { $_.Trim() }).Count
if ($count -ne $summary.completed_variant_answers) { throw 'Prediction count does not match summary' }
Write-Host "Variant answers: $count/600; complete=$($summary.complete)"
Write-Host "Pilot20 questions: $($summary.pilot20.completed_questions)/20; complete=$($summary.pilot20.complete)"
if (-not $summary.complete) { Write-Host 'A complete pilot has 60/600 records; dev200 completion requires 600/600. Exports refresh on worker exit.' }
Write-Host "Da tai vao: $Destination"
