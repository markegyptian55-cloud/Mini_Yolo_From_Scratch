# MiniYOLO Crash-Proof Auto-Resuming Training Launcher
# This script monitors training execution and automatically restarts training
# from the latest checkpoint if interrupted by power, system, or process errors.

Clear-Host
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "     🛡️ MINI YOLO CRASH-PROOF AUTOMATIC RESUME LAUNCHER     " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$attempt = 1

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "`n[$timestamp] 🚀 Run Session #$attempt: Starting / Resuming MiniYOLO Training..." -ForegroundColor Yellow
    
    python -m src.train
    
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host "`n🎉 Training completed all target epochs successfully with exit code 0!" -ForegroundColor Green
        Write-Host "Session finished safely." -ForegroundColor Green
        break
    } else {
        Write-Host "`n⚠️ Warning: Training process stopped unexpectedly (Exit Code: $exitCode)." -ForegroundColor Red
        Write-Host "🔄 Restarting and resuming automatically from the last checkpoint in 10 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        $attempt++
    }
}
