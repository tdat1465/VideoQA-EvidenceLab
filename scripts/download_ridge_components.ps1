param(
    [string]$SshTarget = 'lnthanh03@172.29.74.81',
    [string]$Destination = 'E:\Codex\Documents\ChatGPT\KLTN\outputs\ridge-components-results-' + [DateTime]::Now.ToString('yyyyMMdd-HHmmss')
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Destination) { throw "Destination already exists: $Destination" }
New-Item -ItemType Directory -Path $Destination | Out-Null
$logs = Join-Path $Destination 'logs'
New-Item -ItemType Directory -Path $logs | Out-Null
$repo = '/media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1'
$sources = @("${SshTarget}:${repo}/logs/ridge-cmp-*.out", "${SshTarget}:${repo}/logs/ridge-cmp-*.err", "${SshTarget}:${repo}/logs/ridge-components-*.json")
& scp -p -o StrictHostKeyChecking=yes -o ConnectTimeout=15 @sources $logs
if ($LASTEXITCODE -ne 0) { throw "Incomplete logs; keep files in $logs" }
$remote = '/media/lnthanh03/DatHa/runs/videoqa/lvb-ridge-components-v1-dev200'
$local = Join-Path $Destination 'lvb-ridge-components-v1-dev200'
New-Item -ItemType Directory -Path $local | Out-Null
$files = foreach ($name in @('summary.json','contract.json','manifest.json','predictions.jsonl','environment.latest.txt','gpu.latest.txt')) { "${SshTarget}:${remote}/${name}" }
& scp -p -o StrictHostKeyChecking=yes -o ConnectTimeout=15 @files $local
if ($LASTEXITCODE -ne 0) { throw "Incomplete exports; logs retained in $logs" }
& scp -r -p -o StrictHostKeyChecking=yes -o ConnectTimeout=15 "${SshTarget}:${remote}/curves" $local
if ($LASTEXITCODE -ne 0) { throw "Incomplete curves/scores; exports retained in $local" }
$summary = Get-Content -LiteralPath (Join-Path $local 'summary.json') -Raw | ConvertFrom-Json
$count = @(Get-Content -LiteralPath (Join-Path $local 'predictions.jsonl') | Where-Object { $_.Trim() }).Count
if ($count -ne $summary.completed_variant_answers) { throw 'Prediction count does not match summary' }
Write-Host "Variant answers: $count/400; complete=$($summary.complete)"
if (-not $summary.complete) { Write-Host 'Pilot or partial export. Files are refreshed when the worker exits.' }
Write-Host "Da tai vao: $Destination"
