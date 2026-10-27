# Windows PowerShell 5.1/7 fixture. No real SSH or network connections.
param([Parameter(Mandatory=$true)][string]$WorkRoot)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$script=Join-Path $repoRoot 'scripts/download_ridge_event_full.ps1'
$root=Join-Path $WorkRoot ([Guid]::NewGuid().ToString('N'))
$target=Join-Path $root 'target with spaces'
$reuse=Join-Path $root 'reuse'
$server=Join-Path $root 'server'
$run='lvb-ridge-event-coverage-full1337-v1'
foreach($path in @($target,$reuse,$server)){[void](New-Item -ItemType Directory -Path $path -Force)}
$content=[ordered]@{}
$content["$run/summary.json"]='{"completed_variant_answers":1,"complete":false,"completed_questions":0,"target_variant_answers":6685}'
$content["$run/contract.json"]='{"contract":1}'
$content["$run/manifest.json"]='{"ids":["q1"]}'
$content["$run/predictions.jsonl"]='{"id":"q1"}'
$content['logs/ridge-evtfull-123.out']='complete'
$content["$run/phases-123.log"]='complete'
for($i=1;$i -le 42;$i++){
    $name=$i.ToString('x64')+'.json'
    $content["$run/curves/$name"]='curve-'+$i
}
$content["$run/scores/"+('a'*64)+'.json']='score-a'
$content["$run/scores/"+('b'*64)+'.json']='score-b'
$global:queryInventory=@()
$global:queryServer=@{}
foreach($rel in $content.Keys){
    $path=Join-Path $server $rel
    [void](New-Item -ItemType Directory -Path (Split-Path -Parent $path) -Force)
    [IO.File]::WriteAllText($path,$content[$rel],[Text.UTF8Encoding]::new($false))
    $remote=if($rel.StartsWith('logs/')){'/media/lnthanh03/DatHa/code/VideoQA-EvidenceLab-ridge-v1/'+$rel}else{'/media/lnthanh03/DatHa/runs/videoqa/'+$rel}
    $global:queryServer[$remote]=$path
    $global:queryInventory+=[pscustomobject]@{relative=$rel;remote=$remote;bytes=(Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path).Hash.ToLowerInvariant()}
}
foreach($pair in @(@("$run/summary.json",$target),@("$run/manifest.json",$reuse))){
    $path=Join-Path $pair[1] $pair[0]
    [void](New-Item -ItemType Directory -Path (Split-Path -Parent $path) -Force)
    Copy-Item -LiteralPath (Join-Path $server $pair[0]) -Destination $path
}
$contract=Join-Path $target "$run/contract.json"
[IO.File]::WriteAllText($contract,('x'*$content["$run/contract.json"].Length),[Text.UTF8Encoding]::new($false))
$global:querySftpCalls=0
$global:queryDownloads=@()
$global:queryInterrupt=$true
$global:querySkip=$false
function ssh {
    $null=@($input)
    $global:LASTEXITCODE=0
    '__RIDGE_RESUME_INVENTORY__'+(@{schema=1;files=$global:queryInventory}|ConvertTo-Json -Depth 6 -Compress)
}
function scp { throw 'Regression: scp must not be used for per-file transfers.' }
function sftp {
    $global:querySftpCalls++
    $commands=@($input)
    if($commands[-1] -ne 'bye'){throw 'SFTP session has no explicit end command'}
    foreach($command in $commands){
        if($command -eq 'bye'){continue}
        if($command -notmatch '^get -p "([^"]+)" "([^"]+)"$'){throw "Unexpected command: $command"}
        $remote=$Matches[1];$local=$Matches[2]
        if($global:querySkip){continue}
        Copy-Item -LiteralPath $global:queryServer[$remote] -Destination $local -Force
        $global:queryDownloads+=$remote
        if($global:queryInterrupt){$global:queryInterrupt=$false;$global:LASTEXITCODE=1;return}
    }
    $global:LASTEXITCODE=0
}
$stopped=$false
try{& $script -SshTarget test@mock-host -Destination $target -ReuseFrom $reuse}
catch{if($_.Exception.Message -like 'Download interrupted*'){$stopped=$true}else{throw}}
if(-not $stopped){throw 'Expected a simulated interruption'}
& $script -SshTarget test@mock-host -Destination $target -ReuseFrom $reuse
$expected=$global:queryInventory.Count-2
if($global:queryDownloads.Count -ne $expected){throw 'Incorrect download count or redownloaded complete files'}
if(@($global:queryDownloads|Select-Object -Unique).Count -ne $expected){throw 'A complete file was transferred twice'}
if($global:querySftpCalls -ne 2){throw 'Expected one SFTP process for each interrupted/successful attempt'}
& $script -SshTarget test@mock-host -Destination $target -ReuseFrom $reuse
if($global:querySftpCalls -ne 2){throw 'A completed destination must not create an SFTP session'}
# An interactive SFTP client may exit zero even when a get failed. Hash checks
# must still reject a missing/corrupt file.
[IO.File]::WriteAllText($contract,'partial',[Text.UTF8Encoding]::new($false))
$global:querySkip=$true;$rejected=$false
try{& $script -SshTarget test@mock-host -Destination $target -ReuseFrom $reuse}
catch{if($_.Exception.Message -like 'Missing file or checksum mismatch*'){$rejected=$true}else{throw}}
if(-not $rejected){throw 'Must reject failed file even when SFTP exits zero'}
$global:querySkip=$false
& $script -SshTarget test@mock-host -Destination $target -ReuseFrom $reuse
foreach($entry in $global:queryInventory){
    if((Get-FileHash -LiteralPath (Join-Path $target $entry.relative)).Hash.ToLowerInvariant() -ne $entry.sha256){throw 'Final checksum mismatch'}
}
@{status='passed';server_files=$global:queryInventory.Count;single_sftp_for_over30_files=$true;interruption_resume=$true;no_transfer_on_complete=$true;same_size_corruption=$true;zero_exit_missing_file_rejected=$true;real_network_connections=0}|ConvertTo-Json
