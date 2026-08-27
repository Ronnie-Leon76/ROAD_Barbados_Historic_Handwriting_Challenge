<#
.SYNOPSIS
    Launch the pipeline as a detached background job (Windows).
.EXAMPLE
    .\run_background.ps1 --folds 0 1 2 3 4 --hf-repo me/road-htr
.DESCRIPTION
    Any arguments are forwarded to run_pipeline.py. Prints the PID and log path,
    then returns immediately. Email arrives on checkpoint uploads, GRPO verdicts,
    completion and failure.
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path run_pipeline.py)) {
    Write-Host "run_pipeline.py missing - generating it"
    python build/build_script.py
}

New-Item -ItemType Directory -Force -Path logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = "logs/console_$stamp.out"

Write-Host "checking the notification channel ..."
python build/check_notifications.py

$proc = Start-Process -FilePath "python" `
    -ArgumentList (@("-u", "run_pipeline.py") + $args) `
    -RedirectStandardOutput $out -RedirectStandardError "logs/console_$stamp.err" `
    -NoNewWindow -PassThru
$proc.Id | Out-File -Encoding ascii logs/run.pid

Write-Host ""
Write-Host "started pid $($proc.Id)"
Write-Host "  console log   $out"
Write-Host "  follow        Get-Content $out -Wait"
Write-Host "  metrics       tensorboard --logdir artifacts/runs --port 6006"
Write-Host "  stop          Stop-Process -Id (Get-Content logs/run.pid)"
