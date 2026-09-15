$ErrorActionPreference = "Continue"
$repoPath = "C:\Users\admin\Desktop\FedLLM-Factory-main\FedLLM-Factory-main"
$pythonPath = "D:\sofrware\anaconda3\envs\wzm\python.exe"
$logDir = Join-Path $repoPath "results\stage1\paper_seed42_t768\logs"
$statePath = Join-Path $logDir "serial_state.txt"

function Set-RunState([string]$state) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Set-Content -LiteralPath $statePath -Value "$timestamp $state" -Encoding utf8
}

Set-Location -LiteralPath $repoPath
$env:PYTHONUNBUFFERED = "1"

Set-RunState "centralized_running"
& $pythonPath -X utf8 run_stage1.py --method centralized --run-name paper_seed42_t768 *>> (Join-Path $logDir "centralized.log")
$centralizedExit = $LASTEXITCODE
Set-Content -LiteralPath (Join-Path $logDir "centralized.exitcode") -Value $centralizedExit -Encoding ascii
if ($centralizedExit -ne 0) {
    Set-RunState "centralized_failed exit_code=$centralizedExit"
    exit $centralizedExit
}

Set-RunState "centralized_complete local_running"
& $pythonPath -X utf8 run_stage1.py --method local --run-name paper_seed42_t768 *>> (Join-Path $logDir "local.log")
$localExit = $LASTEXITCODE
Set-Content -LiteralPath (Join-Path $logDir "local.exitcode") -Value $localExit -Encoding ascii
if ($localExit -ne 0) {
    Set-RunState "local_failed exit_code=$localExit"
    exit $localExit
}

Set-RunState "all_complete"
