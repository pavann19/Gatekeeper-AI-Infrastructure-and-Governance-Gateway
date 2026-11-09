param(
    [Parameter(Mandatory=$true)][string]$Run,
    [Parameter(Mandatory=$true)][string]$Image
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskRun = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $Run))
if (Test-Path -LiteralPath $taskRun) { throw 'Evidence directory must be new.' }
New-Item -ItemType Directory -Path $taskRun | Out-Null
$fixtureId = [guid]::NewGuid().ToString('N').Substring(0,16)
$fixtureName = 'gk-synthetic-' + $fixtureId
$fixtureDatabase = 'gk_synthetic_' + $fixtureId
$fixtureOwned = $false
$previousFixtureUrl = $env:GATEKEEPER_TEST_PG_URL
$record = @{
    started_utc = [DateTime]::UtcNow.ToString('o')
    container = $fixtureName
    result = 'RUNNING'
    scope = 'Owned loopback fixture; tmpfs data; 512 MiB and 1 CPU; no existing resources used'
}
try {
    $fixtureImage = (& docker image inspect $Image --format '{{.Id}}').Trim()
    if ($LASTEXITCODE -ne 0 -or -not $fixtureImage.StartsWith('sha256:')) { throw 'Existing local image required.' }
    $record.image = $fixtureImage
    & docker run --detach --pull=never --name $fixtureName --label "gatekeeper.validation.fixture=$fixtureId" `
        --memory=512m --cpus=1 --pids-limit=128 --tmpfs '/tmp:rw,nosuid,size=256m' `
        --tmpfs '/var/run/postgresql:rw,nosuid,mode=1777' --publish '127.0.0.1::5432' `
        --env 'PGDATA=/tmp/pgdata' --env 'POSTGRES_USER=gk_test' --env 'POSTGRES_PASSWORD=synthetic-test-only' `
        --env "POSTGRES_DB=$fixtureDatabase" $fixtureImage | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not create owned fixture.' }
    $fixtureOwned = $true
    $fixtureReady = $false
    for ($attempt=0; $attempt -lt 40; $attempt++) {
        & docker exec $fixtureName pg_isready -U gk_test -d $fixtureDatabase 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $fixtureReady=$true; break }
        Start-Sleep -Milliseconds 250
    }
    if (-not $fixtureReady) { throw 'Owned fixture did not become ready.' }
    $fixtureInspect = (& docker inspect $fixtureName | ConvertFrom-Json)[0]
    $binding = $fixtureInspect.NetworkSettings.Ports.'5432/tcp'[0]
    if ($binding.HostIp -ne '127.0.0.1') { throw 'Fixture must bind only loopback.' }
    $env:GATEKEEPER_TEST_PG_URL = "postgresql://gk_test:synthetic-test-only@127.0.0.1:$($binding.HostPort)/$fixtureDatabase"
    & (Join-Path $taskRoot '.venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'run_isolated_tests.py') `
        --run (Join-Path $taskRun 'contracts') --postgres-fixture --integration tests/test_postgres_state_integration.py `
        *> (Join-Path $taskRun 'contracts-command.log')
    $record.exit_code = $LASTEXITCODE
    $record.result = if ($LASTEXITCODE -eq 0) { 'PASS' } else { 'FAIL' }
    if ($record.result -eq 'PASS') {
        # Restore only this fixture's quiescent synthetic database into a new
        # database in the same owned container; never reuse a host volume.
        $fingerprintQuery = "SELECT (SELECT count(*) FROM gk_audit_events)::text || ':' || " +
            "coalesce((SELECT string_agg(event_hash::text,',' ORDER BY sequence) FROM gk_audit_events),'') || ':' || " +
            "(SELECT count(*) FROM gk_reviews)::text || ':' || " +
            "coalesce((SELECT string_agg(md5(payload::text),',' ORDER BY review_id) FROM gk_reviews),'')"
        $beforeRestore = & docker exec $fixtureName psql -U gk_test -d $fixtureDatabase -At -c $fingerprintQuery
        if ($LASTEXITCODE -ne 0) { throw 'Source fingerprint failed.' }
        $backupSql = & docker exec $fixtureName pg_dump -U gk_test --no-owner --no-privileges $fixtureDatabase
        if ($LASTEXITCODE -ne 0) { throw 'Fixture backup failed.' }
        $backupSql | Set-Content -Encoding utf8 (Join-Path $taskRun 'synthetic-backup.sql')
        $restoreDatabase = $fixtureDatabase + '_restore'
        $restoreWatch = [System.Diagnostics.Stopwatch]::StartNew()
        & docker exec $fixtureName createdb -U gk_test $restoreDatabase
        if ($LASTEXITCODE -ne 0) { throw 'Could not create fresh restore database.' }
        $backupSql | & docker exec -i $fixtureName psql -U gk_test -d $restoreDatabase --set ON_ERROR_STOP=1 `
            *> (Join-Path $taskRun 'restore.log')
        if ($LASTEXITCODE -ne 0) { throw 'Fixture restore failed.' }
        $afterRestore = & docker exec $fixtureName psql -U gk_test -d $restoreDatabase -At -c $fingerprintQuery
        if ($LASTEXITCODE -ne 0 -or ($beforeRestore -join '') -ne ($afterRestore -join '')) {
            throw 'Restored audit/review fingerprints differ.'
        }
        $restoreWatch.Stop()
        $record.restore = @{
            status = 'PASS'
            elapsed_seconds = $restoreWatch.Elapsed.TotalSeconds
            quiescent_fixture_rows_lost = 0
            scope = 'Fresh DB in the same owned fixture; not production RTO/RPO or failover acceptance'
        }
    }
}
catch {
    $record.result = 'INCOMPLETE'
    $record.failure = $_.Exception.Message
}
finally {
    $env:GATEKEEPER_TEST_PG_URL = $previousFixtureUrl
    if ($fixtureOwned) {
        $fixtureInspect = (& docker inspect $fixtureName | ConvertFrom-Json)[0]
        if ($fixtureInspect.Config.Labels.'gatekeeper.validation.fixture' -ne $fixtureId) {
            throw 'Refusing cleanup: ownership label differs.'
        }
        & docker logs $fixtureName *> (Join-Path $taskRun 'postgres.log')
        & docker rm --force --volumes $fixtureName | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Owned fixture cleanup failed.' }
        $record.owned_fixture_removed = $true
    }
    $record.finished_utc = [DateTime]::UtcNow.ToString('o')
    $record | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 (Join-Path $taskRun 'receipt.json')
}
$record | ConvertTo-Json -Depth 6
if ($record.result -ne 'PASS') { exit 1 }
