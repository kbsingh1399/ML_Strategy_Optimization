# Launch Chrome in Remote Debugging Mode on port 19022 for Arena.ai Automation
$chromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$userDataDir = "$env:LOCALAPPDATA\Google\Chrome\User Data_Arena"
$targetUrl = "https://arena.ai/agent/019fbc51-76db-79e8-b0d2-c8da2966516a"

Write-Host "[Autonomous Loop] Starting Chrome on port 19022 for Arena Chat Agent..." -ForegroundColor Green
Start-Process $chromePath -ArgumentList "--remote-debugging-port=19022", "--user-data-dir=`"$userDataDir`"", "$targetUrl"
