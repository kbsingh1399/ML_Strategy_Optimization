# Launch Chrome in Remote Debugging Mode on port 19022 for Arena.ai Automation
$chromePath = "chrome.exe"
$userDataDir = "$env:LOCALAPPDATA\Google\Chrome\User Data_Arena"

Write-Host "[Autonomous Loop] Starting Chrome on port 19022..." -ForegroundColor Green
Start-Process $chromePath -ArgumentList "--remote-debugging-port=19022 --user-data-dir=`"$userDataDir`""
