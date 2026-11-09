param([Parameter(Mandatory=$true)][string]$Run, [Parameter(Mandatory=$true)][string]$Image)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskRun = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $Run))
if (Test-Path -LiteralPath $taskRun) { throw 'Evidence directory must be new.' }
New-Item -ItemType Directory -Path $taskRun | Out-Null
$fixtureId = [guid]::NewGuid().ToString('N').Substring(0,16)
$fixtureName = 'gk-redis-synthetic-' + $fixtureId
$fixtureOwned = $false
$previousUrl = $env:GATEKEEPER_REDIS_INTEGRATION_URL
$record = @{started_utc=[DateTime]::UtcNow.ToString('o');container=$fixtureName;result='RUNNING';scope='Owned loopback Redis fixture, 128 MiB / 1 CPU, no existing volumes'}
try {
    $fixtureImage = (& docker image inspect $Image --format '{{.Id}}').Trim()
    if ($LASTEXITCODE -ne 0 -or -not $fixtureImage.StartsWith('sha256:')) { throw 'Existing local image required.' }
    $record.image = $fixtureImage
    & docker run --detach --pull=never --name $fixtureName --label "gatekeeper.validation.fixture=$fixtureId" `
        --memory=128m --cpus=1 --pids-limit=64 --publish '127.0.0.1::6379' $fixtureImage `
        redis-server --appendonly no --maxmemory 64mb --maxmemory-policy noeviction --requirepass synthetic-test-only | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not create owned Redis fixture.' }
    $fixtureOwned = $true
    for ($attempt=0; $attempt -lt 40; $attempt++) {
        & docker exec $fixtureName redis-cli -a synthetic-test-only ping 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Milliseconds 250
    }
    $fixtureInspect = (& docker inspect $fixtureName | ConvertFrom-Json)[0]
    $binding = $fixtureInspect.NetworkSettings.Ports.'6379/tcp'[0]
    if ($binding.HostIp -ne '127.0.0.1') { throw 'Fixture must bind only loopback.' }
    $env:GATEKEEPER_REDIS_INTEGRATION_URL = "redis://:synthetic-test-only@127.0.0.1:$($binding.HostPort)/0"
    $python = Join-Path $taskRoot '.venv\Scripts\python.exe'
    & $python -B (Join-Path $PSScriptRoot 'run_isolated_tests.py') --run (Join-Path $taskRun 'healthy') `
        --redis-fixture tests/test_live_redis_acceptance.py *> (Join-Path $taskRun 'healthy-command.log')
    if ($LASTEXITCODE -ne 0) { throw 'Healthy Redis contracts failed.' }
    $fixtureInspect = (& docker inspect $fixtureName | ConvertFrom-Json)[0]
    if ($fixtureInspect.Config.Labels.'gatekeeper.validation.fixture' -ne $fixtureId) { throw 'Ownership label differs.' }
    & docker pause $fixtureName | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not pause owned fixture.' }
    & $python -B (Join-Path $PSScriptRoot 'run_isolated_tests.py') --run (Join-Path $taskRun 'outage') `
        --redis-outage-fixture tests/test_live_redis_outage.py *> (Join-Path $taskRun 'outage-command.log')
    $record.exit_code = $LASTEXITCODE
    $record.result = if ($LASTEXITCODE -eq 0) { 'PASS' } else { 'FAIL' }
}
catch { $record.result='INCOMPLETE'; $record.failure=$_.Exception.Message }
finally {
    $env:GATEKEEPER_REDIS_INTEGRATION_URL = $previousUrl
    if ($fixtureOwned) {
        $fixtureInspect = (& docker inspect $fixtureName | ConvertFrom-Json)[0]
        if ($fixtureInspect.Config.Labels.'gatekeeper.validation.fixture' -ne $fixtureId) { throw 'Refusing cleanup: ownership differs.' }
        if ($fixtureInspect.State.Paused) { & docker unpause $fixtureName | Out-Null }
        & docker logs $fixtureName *> (Join-Path $taskRun 'redis.log')
        & docker rm --force --volumes $fixtureName | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Owned Redis fixture cleanup failed.' }
        $record.owned_fixture_removed=$true
    }
    $record.finished_utc=[DateTime]::UtcNow.ToString('o')
    $record | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 (Join-Path $taskRun 'receipt.json')
}
$record | ConvertTo-Json -Depth 6
if ($record.result -ne 'PASS') { exit 1 }
