param([Parameter(Mandatory=$true)][string]$WorkRoot)
$ErrorActionPreference='Stop'
$script=Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts/download_ridge_decode_policy.ps1'
$target=Join-Path $WorkRoot ([Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $target -Force)
$global:driftPayload=@{schema='ridge-decode-policy-diagnostic-v1';target='longvideobench:val:o2F-N42Ufo4_2';evaluation_results=$false;complete=$true;snapshot_unchanged=$true;reference_unchanged=$true;cases=@{a=@{};b=@{};c=@{};d=@{};e=@{};f=@{};g=@{};h=@{}}}
$global:driftScpCalls=0
$global:driftDownloadFail=$false
function scp {
    $global:driftScpCalls++
    if($global:driftDownloadFail){$global:LASTEXITCODE=1;return}
    [IO.File]::WriteAllText($args[-1],($global:driftPayload|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
    $global:LASTEXITCODE=0
}
& $script -JobId 99999 -SshTarget test@mock -Destination $target
$path=Join-Path $target 'decode-99999.json'
if(-not (Test-Path -LiteralPath $path)){throw 'Missing final report'}
$before=(Get-FileHash -LiteralPath $path).Hash
if($global:driftScpCalls -ne 1){throw 'Expected exactly one SCP connection'}
$global:driftPayload.schema='unexpected'
$rejected=$false
try{& $script -JobId 99999 -SshTarget test@mock -Destination $target}catch{$rejected=$true}
if(-not $rejected -or (Get-FileHash -LiteralPath $path).Hash -ne $before){throw 'Invalid payload overwrote a valid report'}
$global:driftDownloadFail=$true
$rejected=$false
try{& $script -JobId 99999 -SshTarget test@mock -Destination $target}catch{$rejected=$true}
if(-not $rejected -or (Get-FileHash -LiteralPath $path).Hash -ne $before){throw 'Failed transfer overwrote a valid report'}
@{status='passed';real_network_connections=0;one_scp_for_single_report=$true;invalid_payload_keeps_old_report=$true;failed_transfer_keeps_old_report=$true}|ConvertTo-Json
